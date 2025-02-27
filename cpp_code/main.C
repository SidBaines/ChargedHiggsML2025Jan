// ./execute MCbase.config /data/atlas/HplusWh/20241021_RawNtuples/user.rhulsken.mc16_13TeV.510124.MGPy8EGNNPDF30_Hp_H3000_Whbb.TOPQ1.e8448s3126r9364p4514.Nominal_v0_1l_out_root /data/atlas/HplusWh/20241211_tmp
#include "main/EventLoop.h"
#include "TApplication.h"
#include <TSystemDirectory.h>
#include <TList.h>
#include <TCollection.h>
#include <TChain.h>
#include <TFile.h>
#include <TH1F.h>
#include <TTree.h>
#include <TString.h>
#include <TRegexp.h> 
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <ctime>
#include <tuple>
#include "utilis/configparser.h"
//#include <boost/filesystem.hpp>
std::tuple<float, float, float> CalcNormalisationFactor(int dsid, std::string Samplename, bool debugMode = false, bool dataMode = false);
float CalcLumiFactor(std::string Samplename, bool debugMode = false, bool dataMode = false);

int main(int argc, char **argv)
{
  //clock_t begin = clock();
  if (argc < 4){
    std::cout << "Needs 3 arguments. Syntax is: " << std::endl << "\t\t ./execute configFilePath inputRootFile outputDirectory" << std::endl;
    return 1;
  }
  TApplication theApp("evsel", &argc, argv);
  auto config = parseConfig(theApp.Argv(1));

  std::string FullFileName = theApp.Argv(2);
  TString OUTPUTDIR = theApp.Argv(3);

  std::size_t found_penultimate_delim;
  found_penultimate_delim = FullFileName.find_last_of("/");
  std::string SampleName = FullFileName.substr(found_penultimate_delim+1);
  std::string stdpath = FullFileName.substr(0, found_penultimate_delim+1);
  std::cout << "SampleName: " << SampleName << std::endl;
  std::cout << "stdpath: " << stdpath << std::endl;

  TString WP = config["WP"];
  int EventReadout = stoi(config["EventReadout"]);
  bool batchMode = atoi(config["batchMode"].c_str());
  Long64_t minEvent = stoi(config["MinimumEvent"]);
  Long64_t maxEvent = stoi(config["MaximumEvent"]);
  TString TreeName = TString("nominal_Loose");

  bool debugMode = (config["Debug_Mode"] == "Enable" || config["Debug_Mode"] == "enable" );
  bool dataMode = !(config["Has_weights"] == "Enable" || config["Has_weights"] == "enable" ); // TODO Probably ought to rename the config file variable here, to something like 'data_rather_than_mc'

  // Set the luminosity factor based upon the AMI tags.
  float luminosity_factor = CalcLumiFactor(SampleName, debugMode, dataMode);

  TString path = TString(stdpath);
  path += TString(SampleName) + "/";
  TString OutFileName = SampleName;

  // TODO put in something to print out a helpful degub message when the filename isn't a root file or isn't even a file (currently gives terminate called after throwing an instance of 'std::invalid_argument' what (): stoi )

  if (!batchMode)
  {
    OutFileName = OUTPUTDIR + "/" + SampleName;
  }

  // system("mkdir " + OUTPUTDIR + "/" + SampleName);
  
  std::string file_extention = ".root";
  std::string ex_OutFileName = std::string(OutFileName);
  //std::cout << ex_OutFileName << std::endl;
  ex_OutFileName.replace(ex_OutFileName.end() - 5, ex_OutFileName.end(), file_extention);
  //std::cout << ex_OutFileName << std::endl;

  TFile *outfile = TFile::Open(ex_OutFileName.c_str(), "RECREATE");
  //std::cout << "here 3" << std::endl;
  TChain *mych_data = new TChain(TreeName);
  mych_data->Add(path + "*.root");
  Long64_t nentries = mych_data->GetEntries();
  std::cout << "\nnentires = " << nentries << " for file " << path << "\n";

  // Calculate and output the normalisation factor (by which we must divide)
  uint dsid_int = 0;
  if (debugMode) std::cout << "dsid_int initialized to: " << dsid_int << std::endl;
  if (!dataMode)
  {
    mych_data->SetBranchAddress("mcChannelNumber", &dsid_int);
    if (debugMode) std::cout << "dsid_int updated to: " << dsid_int << std::endl;
    mych_data->GetEntry(0);
    if (debugMode) std::cout << "dsid_int updated to: " << dsid_int << std::endl;
  }

  //std::cout << "ex_OutFileName: " << ex_OutFileName << std::endl;
  // EventLoop *eventLoop = new EventLoop(mych_data);
  EventLoop *eventLoop = new EventLoop(mych_data, TreeName, ex_OutFileName, config);
  eventLoop->SetDebugMode(debugMode);
  if (nentries == 0)
  {
    std::cout << "nentries == 0 so skipping \n";
    eventLoop->WriteTreeToFile(outfile);
    outfile->Close();
    return 0;
  }
  if (mych_data == 0)
  {
    std::cout << "mych_data == 0 \n";
    return 0;
  }
  float xsec, kfac, sumOfMCGenWeights;
  std::tie(xsec, kfac, sumOfMCGenWeights) = CalcNormalisationFactor(dsid_int, SampleName, debugMode, dataMode);
  eventLoop->SetNormFactor(xsec, kfac, sumOfMCGenWeights);
  if (!dataMode) eventLoop->SetLumiFactor(luminosity_factor);
  //std::cout << "Here 6" << std::endl;
  Long64_t nbytes = 0, nb = 0;

  if (minEvent == 0 && maxEvent == 0)
  {
    minEvent = 0;
    maxEvent = nentries;
  }
  else if (maxEvent > nentries)
  {
    maxEvent = nentries;
  }
  if (debugMode) maxEvent = minEvent + std::stoi(config["Num_Debug_Events"]);
  if (maxEvent > nentries) maxEvent = nentries; // Make sure the debug case doesn't accidentally break the code by asking for too many events

  std::cout << "Start Event : " << minEvent << "\n"
            << "End Event : " << maxEvent << "\n";
  
  bool LowLevelPass;
  for (Long64_t jentry = minEvent; jentry < maxEvent; jentry++)
  {
    if (debugMode) std::cout << "Loading entry: " << jentry << std::endl;
    //if (jentry==7592160) continue; // Skip this event for one particular file (ttbar_lep) because I think there's an error (I think there is 1 fat jet, but it doesn't have an associated ljet_Xbb2020v3_Top value)
    //if (jentry==10827225) continue; // Skip this event for one particular file (ttbar_lep) because I think there's an error (I think there is 1 fat jet, but it doesn't have an associated ljet_Xbb2020v3_Top value)
    Long64_t ientry = eventLoop->LoadTree(jentry);

    if (ientry < 0 || !mych_data)
      break;

    nb = mych_data->GetEntry(jentry);
    nbytes += nb;
    if (EventReadout != 0)
    {
      if (jentry % EventReadout == 0)
      {
        std::cout << "Processing " << jentry << " events!!!"
                  << "\n";
        std::cout << "Data : " << nbytes << " Bytes"
                  << "\n"
                  << "Data : " << nbytes * 0.000000001 << " Gigabytes"
                  << "\n";
      }
    }
    if (jentry % 100000 == 0){
      std::cout << "Processing entry: " << jentry << std::endl;
    }
    // if (jentry == 2000000){
    //   break;
    // }
    // std::cout << "Processing entry: " << jentry << "     ";
    if ((jentry == 156462) && (SampleName == "user.rhulsken.mc16_13TeV.410470.PhPy8EG_ttbar_lep.TOPQ1.e6337s3126r10724p4514.Nominal_v0_1l_out_root")) continue;
    if ((jentry == 5366719) && (SampleName == "user.rhulsken.mc16_13TeV.410470.PhPy8EG_ttbar_lep.TOPQ1.e6337s3126r9364p4514.Nominal_v0_1l_out_root")) continue;
    if ((jentry == 7729035) && (SampleName == "user.rhulsken.mc16_13TeV.410470.PhPy8EG_ttbar_lep.TOPQ1.e6337s3126r9364p4514.Nominal_v0_1l_out_root")) continue;
    
    LowLevelPass = eventLoop->LowLevel_Loop();
    if (!LowLevelPass) continue;
    eventLoop->Loop();
    // Fill the output tree
    if ((eventLoop->WriteAllEvents) || (eventLoop->selection_category==0) || (eventLoop->selection_category==3) || (eventLoop->selection_category==8) || (eventLoop->selection_category==9) || (eventLoop->selection_category==10)){
      eventLoop->output_tree->Fill();
    }
  } // end loop over jentry (events)
  // std::cout << "Finished PreWriteTreeToFile" << std::endl;
  // eventLoop->WriteTreeToFile(outfile);
  // std::cout << "Finished WriteTreeToFile" << std::endl;
  delete eventLoop;
  // std::cout << "Finished eventLoop" << std::endl;
  delete mych_data;
  // std::cout << "Finished mych_data" << std::endl;
  // outfile->Close();
  // std::cout << "Finished outfile->Close()" << std::endl;
  // std::cout << "Finished looping over the events" << std::endl;
  return 0;
}


std::tuple<float, float, float> CalcNormalisationFactor(int dsid_int, std::string Samplename, bool debugMode, bool dataMode){
  // TODO currently I'm using both the TString and std::string versions of the filename as I'm using old code snippets that I know work; could/should fix this later
  #include "main/xsec.h"
  #include "main/xSecSumOfWeightsFactors.h"
  if (debugMode) std::cout << "Entering CalcNormalisationFactor" << std::endl;
  if (debugMode) std::cout << "\t" << "NB: This assumes that all entries in this file will have the same DSID (set it only once)" << std::endl;
  float sumOfMCGenWeights = 1;
  float xsec = 1;
  float kfac = 1;
  if (!dataMode)
  {
    if (debugMode) std::cout << dsid_int << std::endl;
    sumOfMCGenWeights = 0;
    std::string MCDataPeriod;
    if (Samplename.find("r9364") != std::string::npos) MCDataPeriod = "MC16a";
    if (Samplename.find("r10201") != std::string::npos) MCDataPeriod = "MC16d";
    if (Samplename.find("r10724") != std::string::npos) MCDataPeriod = "MC16e";
    if (Samplename.find("MC16a") != std::string::npos) MCDataPeriod = "MC16a";
    if (Samplename.find("MC16d") != std::string::npos) MCDataPeriod = "MC16d";
    if (Samplename.find("MC16e") != std::string::npos) MCDataPeriod = "MC16e";
    sumOfMCGenWeights = xSecSumOfWeightsFactors[MCDataPeriod][dsid_int];
    assert(("xsec.h map 'dsid_xsec' doesn't contain an entry for " + std::to_string(dsid_int), dsid_xsec.count(dsid_int)));
    xsec = dsid_xsec[dsid_int];
    assert(("xsec.h map 'dsid_kfac' doesn't contain an entry for " + std::to_string(dsid_int), dsid_kfac.count(dsid_int)));
    kfac = dsid_kfac[dsid_int];
  }
  else{
    if (debugMode) std::cout << "In dataMode so leaving xsec, kfac, sumOfMCGenWeights all equal to 1.0" << std::endl;
  }
  if (debugMode) std::cout << "\t" << "sumOfMCGenWeights: " << sumOfMCGenWeights << std::endl;
  if (debugMode) std::cout << "\t" << "xsec: " << xsec << std::endl;
  if (debugMode) std::cout << "\t" << "kfac: " << kfac << std::endl;
  if (debugMode) std::cout << "Leaving CalcNormalisationFactor" << std::endl;
  return std::make_tuple(xsec, kfac, sumOfMCGenWeights);
}

float CalcLumiFactor(std::string Samplename, bool debugMode, bool dataMode)
{
  if (debugMode) std::cout << "Entering CalcLumiFactor" << std::endl;
  if (debugMode) std::cout << "\t" << "Samplename: " << Samplename << std::endl;
  float luminosity_factor = -1.0;
  if (!dataMode)
  {
    if (Samplename.find("r9364") != std::string::npos) luminosity_factor = 3219.56+32988.1; // r9364 is the AMI tag for MC16a campagin; data15+data16 luminosity is 36207.66 pb^-1
    if (Samplename.find("r10201") != std::string::npos) luminosity_factor = 44307.4; // r10201 is the AMI tag for MC16d campagin; data17 luminosity is 44307.4 pb^-1
    if (Samplename.find("r10724") != std::string::npos) luminosity_factor = 58450.1; // r10724 is the AMI tag for MC16e campagin; data18 luminosity is 58450.1 pb^-1
    if (Samplename.find("MC16a") != std::string::npos) luminosity_factor = 3219.56+32988.1; // MC16a campaign data15+data16 luminosity is 36207.66 pb^-1
    if (Samplename.find("MC16d") != std::string::npos) luminosity_factor = 44307.4; // MC16d campagin data17 luminosity is 44307.4 pb^-1
    if (Samplename.find("MC16e") != std::string::npos) luminosity_factor = 58450.1; // MC16e campagin data18 luminosity is 58450.1 pb^-1
    assert(("luminosity_factor has not been set", luminosity_factor > 0));
    if (debugMode) std::cout << "Luminosity factor has been set to: " << luminosity_factor << std::endl;;
  }
  if (debugMode) std::cout << "Leaving CalcLumiFactor" << std::endl;
  return luminosity_factor;
}