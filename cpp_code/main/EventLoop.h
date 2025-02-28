//////////////////////////////////////////////////////////
// This class has been automatically generated on
// Sun Jan 16 15:31:40 2022 by ROOT version 6.20/06
// from TTree nominal_Loose/tree
// found on file: data/user.adsalvad.mc16_13TeV.504567.MGPy8EGNNPDF30_Hp_H1600_Wh.TOPQ1.e8273a875r10724p4346.HplusWh212171-v3_1l_out_root/user.adsalvad.26529705._000001.out.root
//////////////////////////////////////////////////////////

#ifndef EventLoop_h
#define EventLoop_h

#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>
#include <TRegexp.h> 

// Header file for the classes stored in the TTree if any.
#include "vector"
#include "TLorentzVector.h"
#include "string"
#include <iostream>
#include <fstream>
#include "utilis/NeutrinoBuilder.h"
//#include "TMVA/Reader.h"
#include <unordered_map>
#include <string>
#include <sstream>
#include <iomanip>

#include "lwtnn/parse_json.hh"
#include "lwtnn/LightweightGraph.hh"
// #include <lwtnn/parse_json.hh>
// #include <lwtnn/LightweightGraph.hh>

// #include<map>

// Struct to represent a particle
struct Particle {
    TLorentzVector p4;
    int type;  // 0: electron, 1: muon, 2: neutrino, 3: ljet, 4: sjet
    float tagInfo;  // Tagging score (0 for electron/muon/neutrino)
    int recoInclusion;  // 0: Not included in the reconstructed charged Higgs, 1: incldued as part of SM higgs from H+, 2: included as part of Wqq, 3: included as part of Wlv (this will indeed just be the lepton and neutrino)
    int trueInclusion;  // 0: Not included in the     true      charged Higgs, 1: incldued as part of SM higgs from H+, 2: included as part of Wqq, 3: included as part of Wlv (this will indeed just be the lepton and neutrino)
};

class EventLoop {
public :
   TTree          *fChain;   //!pointer to the analyzed TTree or TChain
   Int_t           fCurrent; //!current Tree number in a TChain
   TTree          *output_tree; // Pointer to the output TTree

// Fixed size dimensions of array or collections stored in the TTree if any.

   // Declaration of leaf types
   // These are the branches in our input L1 ntuple. Note not all the contents of the L1 are included here, nor are all of the variables included here actually used in the code
   std::vector<float>   *mc_generator_weights;
   Float_t         GenFiltHT;
   Float_t         weight_mc;
   Float_t         weight_pileup;
   Float_t         weight_leptonSF;
   Float_t         weight_globalLeptonTriggerSF;
   Float_t         weight_oldTriggerSF;
   Float_t         weight_bTagSF_DL1r_Continuous;
   Float_t         weight_trackjet_bTagSF_DL1r_Continuous;
   Float_t         weight_jvt;
   Float_t         weight_pileup_UP;
   Float_t         weight_pileup_DOWN;
   Float_t         weight_leptonSF_EL_SF_Trigger_UP;
   Float_t         weight_leptonSF_EL_SF_Trigger_DOWN;
   Float_t         weight_leptonSF_EL_SF_Reco_UP;
   Float_t         weight_leptonSF_EL_SF_Reco_DOWN;
   Float_t         weight_leptonSF_EL_SF_ID_UP;
   Float_t         weight_leptonSF_EL_SF_ID_DOWN;
   Float_t         weight_leptonSF_EL_SF_Isol_UP;
   Float_t         weight_leptonSF_EL_SF_Isol_DOWN;
   Float_t         weight_leptonSF_MU_SF_Trigger_STAT_UP;
   Float_t         weight_leptonSF_MU_SF_Trigger_STAT_DOWN;
   Float_t         weight_leptonSF_MU_SF_Trigger_SYST_UP;
   Float_t         weight_leptonSF_MU_SF_Trigger_SYST_DOWN;
   Float_t         weight_leptonSF_MU_SF_ID_STAT_UP;
   Float_t         weight_leptonSF_MU_SF_ID_STAT_DOWN;
   Float_t         weight_leptonSF_MU_SF_ID_SYST_UP;
   Float_t         weight_leptonSF_MU_SF_ID_SYST_DOWN;
   Float_t         weight_leptonSF_MU_SF_ID_STAT_LOWPT_UP;
   Float_t         weight_leptonSF_MU_SF_ID_STAT_LOWPT_DOWN;
   Float_t         weight_leptonSF_MU_SF_ID_SYST_LOWPT_UP;
   Float_t         weight_leptonSF_MU_SF_ID_SYST_LOWPT_DOWN;
   Float_t         weight_leptonSF_MU_SF_Isol_STAT_UP;
   Float_t         weight_leptonSF_MU_SF_Isol_STAT_DOWN;
   Float_t         weight_leptonSF_MU_SF_Isol_SYST_UP;
   Float_t         weight_leptonSF_MU_SF_Isol_SYST_DOWN;
   Float_t         weight_leptonSF_MU_SF_TTVA_STAT_UP;
   Float_t         weight_leptonSF_MU_SF_TTVA_STAT_DOWN;
   Float_t         weight_leptonSF_MU_SF_TTVA_SYST_UP;
   Float_t         weight_leptonSF_MU_SF_TTVA_SYST_DOWN;
   Float_t         weight_globalLeptonTriggerSF_EL_Trigger_UP;
   Float_t         weight_globalLeptonTriggerSF_EL_Trigger_DOWN;
   Float_t         weight_globalLeptonTriggerSF_MU_Trigger_STAT_UP;
   Float_t         weight_globalLeptonTriggerSF_MU_Trigger_STAT_DOWN;
   Float_t         weight_globalLeptonTriggerSF_MU_Trigger_SYST_UP;
   Float_t         weight_globalLeptonTriggerSF_MU_Trigger_SYST_DOWN;
   Float_t         weight_oldTriggerSF_EL_Trigger_UP;
   Float_t         weight_oldTriggerSF_EL_Trigger_DOWN;
   Float_t         weight_oldTriggerSF_MU_Trigger_STAT_UP;
   Float_t         weight_oldTriggerSF_MU_Trigger_STAT_DOWN;
   Float_t         weight_oldTriggerSF_MU_Trigger_SYST_UP;
   Float_t         weight_oldTriggerSF_MU_Trigger_SYST_DOWN;
   Float_t         weight_jvt_UP;
   Float_t         weight_jvt_DOWN;
   std::vector<float>   *weight_bTagSF_DL1r_Continuous_eigenvars_B_up;
   std::vector<float>   *weight_bTagSF_DL1r_Continuous_eigenvars_C_up;
   std::vector<float>   *weight_bTagSF_DL1r_Continuous_eigenvars_Light_up;
   std::vector<float>   *weight_bTagSF_DL1r_Continuous_eigenvars_B_down;
   std::vector<float>   *weight_bTagSF_DL1r_Continuous_eigenvars_C_down;
   std::vector<float>   *weight_bTagSF_DL1r_Continuous_eigenvars_Light_down;
   std::vector<float>   *weight_trackjet_bTagSF_DL1r_Continuous_eigenvars_B_up;
   std::vector<float>   *weight_trackjet_bTagSF_DL1r_Continuous_eigenvars_C_up;
   std::vector<float>   *weight_trackjet_bTagSF_DL1r_Continuous_eigenvars_Light_up;
   std::vector<float>   *weight_trackjet_bTagSF_DL1r_Continuous_eigenvars_B_down;
   std::vector<float>   *weight_trackjet_bTagSF_DL1r_Continuous_eigenvars_C_down;
   std::vector<float>   *weight_trackjet_bTagSF_DL1r_Continuous_eigenvars_Light_down;
   ULong64_t       eventNumber;
   UInt_t          runNumber;
   UInt_t          randomRunNumber;
   UInt_t          mcChannelNumber;
   Float_t         mu;
   Float_t         mu_actual;
   std::vector<float>   *el_pt;
   std::vector<float>   *el_eta;
   std::vector<float>   *el_cl_eta;
   std::vector<float>   *el_phi;
   std::vector<float>   *el_e;
   std::vector<float>   *el_charge;
   std::vector<float>   *el_topoetcone20;
   std::vector<float>   *el_ptvarcone20;
   std::vector<char>    *el_isTight;
   std::vector<char>    *el_CF;
   std::vector<float>   *el_d0sig;
   std::vector<float>   *el_delta_z0_sintheta;
   std::vector<float>   *mu_pt;
   std::vector<float>   *mu_eta;
   std::vector<float>   *mu_phi;
   std::vector<float>   *mu_e;
   std::vector<float>   *mu_charge;
   std::vector<float>   *mu_topoetcone20;
   std::vector<float>   *mu_ptvarcone30;
   std::vector<char>    *mu_isTight;
   std::vector<float>   *mu_d0sig;
   std::vector<float>   *mu_delta_z0_sintheta;
   std::vector<float>   *jet_pt;
   std::vector<float>   *jet_eta;
   std::vector<float>   *jet_phi;
   std::vector<float>   *jet_e;
   std::vector<float>   *jet_jvt;
   std::vector<int>     *jet_truthflav;
   std::vector<int>     *jet_truthflavExtended;
   std::vector<char>    *jet_isbtagged_DL1r_60;
   std::vector<char>    *jet_isbtagged_DL1r_70;
   std::vector<char>    *jet_isbtagged_DL1r_77;
   std::vector<char>    *jet_isbtagged_DL1r_85;
   std::vector<int>     *jet_tagWeightBin_DL1r_Continuous;
   std::vector<float>   *jet_DL1r;
   std::vector<float>   *ljet_pt;
   std::vector<float>   *ljet_eta;
   std::vector<float>   *ljet_phi;
   std::vector<float>   *ljet_e;
   std::vector<float>   *ljet_m;
   std::vector<int>     *ljet_truthLabel;
   std::vector<float>   *tjet_pt;
   std::vector<float>   *tjet_eta;
   std::vector<float>   *tjet_phi;
   std::vector<float>   *tjet_e;
   std::vector<int>     *tjet_tagWeightBin_DL1r_Continuous;
   std::vector<float>   *tjet_DL1r;
   Float_t         met_met;
   Float_t         met_phi;
   //Int_t           ejets_2018_DL1r;
   //Int_t           mujets_2018_DL1r;
   //Int_t           boosted_ljets_ejets_2018_DL1r;
   //Int_t           boosted_ljets_mujets_2018_DL1r;
   Char_t          HLT_mu50;
   Char_t          HLT_mu26_ivarmedium;
   Char_t          HLT_e60_lhmedium_nod0;
   Char_t          HLT_e140_lhloose_nod0;
   Char_t          HLT_e26_lhtight_nod0_ivarloose;
   std::vector<float>   *ljet_C2;
   std::vector<float>   *ljet_D2;
   std::vector<float>   *ljet_Xbb2020v3_Higgs;
   std::vector<float>   *ljet_Xbb2020v3_QCD;
   std::vector<float>   *ljet_Xbb2020v3_Top;
   std::vector<float>   *ljet_muonCorrectedEta;
   std::vector<float>   *ljet_muonCorrectedMass;
   std::vector<float>   *ljet_muonCorrectedPhi;
   std::vector<float>   *ljet_muonCorrectedPt;
   Int_t           HF_Classification;
   Int_t           HF_ClassificationGhost;
   Int_t           HF_SimpleClassification;
   Int_t           HF_SimpleClassificationGhost;
   Int_t           TopHeavyFlavorFilterFlag;
   Int_t           nBTagsTrackJets_DL1r_60;
   Int_t           nBTagsTrackJets_DL1r_70;
   Int_t           nBTagsTrackJets_DL1r_77;
   Int_t           nBTagsTrackJets_DL1r_85;
   Int_t           nBTags_DL1r_60;
   Int_t           nBTags_DL1r_70;
   Int_t           nBTags_DL1r_77;
   Int_t           nBTags_DL1r_85;
   Int_t           nElectrons;
   Int_t           nJets;
   Int_t           nLJets;
   Int_t           nLJets_matched;
   Int_t           nMuons;
   Int_t           nPDFFlavor;
   Int_t           nPrimaryVtx;
   Int_t           nTaus;
   Float_t         Aplanarity_bjets_77;
   Float_t         Aplanarity_bjets_85;
   Float_t         Aplanarity_bjets_Sort4;
   Float_t         Aplanarity_jets;
   Float_t         Centrality_all;
   Float_t         H0_all;
   Float_t         H1_all;
   Float_t         H2_jets;
   Float_t         H4_all;
   Float_t         HT_all;
   Float_t         HT_jets;
   Float_t         Mbb_HiggsMass_77;
   Float_t         Mbb_HiggsMass_85;
   Float_t         Mbb_HiggsMass_Sort4;
   Float_t         Mbb_MaxM_77;
   Float_t         Mbb_MaxM_85;
   Float_t         Mbb_MaxM_Sort4;
   Float_t         Mbb_MaxPt_77;
   Float_t         Mbb_MaxPt_85;
   Float_t         Mbb_MaxPt_Sort4;
   Float_t         Mbb_MinM_77;
   Float_t         Mbb_MinM_85;
   Float_t         Mbb_MinM_Sort4;
   Float_t         Mbb_MindR_77;
   Float_t         Mbb_MindR_85;
   Float_t         Mbb_MindR_Sort4;
   Float_t         Mbj_MaxPt_77;
   Float_t         Mbj_MaxPt_85;
   Float_t         Mbj_MaxPt_Sort4;
   Float_t         Mbj_MindR_77;
   Float_t         Mbj_MindR_85;
   Float_t         Mbj_MindR_Sort4;
   Float_t         Mbj_Wmass_77;
   Float_t         Mbj_Wmass_85;
   Float_t         Mbj_Wmass_Sort4;
   Float_t         Mjj_HiggsMass;
   Float_t         Mjj_MaxPt;
   Float_t         Mjj_MinM;
   Float_t         Mjj_MindR;
   Float_t         Mjjj_MaxPt;
   Float_t         Muu_MindR_77;
   Float_t         Muu_MindR_85;
   Float_t         Muu_MindR_Sort4;
   Float_t         dRbb_MaxM_77;
   Float_t         dRbb_MaxM_85;
   Float_t         dRbb_MaxM_Sort4;
   Float_t         dRbb_MaxPt_77;
   Float_t         dRbb_MaxPt_85;
   Float_t         dRbb_MaxPt_Sort4;
   Float_t         dRbb_MindR_77;
   Float_t         dRbb_MindR_85;
   Float_t         dRbb_MindR_Sort4;
   Float_t         dRbb_avg_77;
   Float_t         dRbb_avg_85;
   Float_t         dRbb_avg_Sort4;
   Float_t         dRbj_Wmass_77;
   Float_t         dRbj_Wmass_85;
   Float_t         dRbj_Wmass_Sort4;
   Float_t         dRlepbb_MindR_77;
   Float_t         dRlepbb_MindR_85;
   Float_t         dRlepbb_MindR_Sort4;
   Float_t         dRlj_MindR;
   Float_t         dRuu_MindR_77;
   Float_t         dRuu_MindR_85;
   Float_t         dRuu_MindR_Sort4;
   Float_t         pT_jet3;
   Float_t         pT_jet5;
   Float_t         pTbb_MindR_77;
   Float_t         pTbb_MindR_85;
   Float_t         pTbb_MindR_Sort4;
   Float_t         pTuu_MindR_77;
   Float_t         pTuu_MindR_85;
   Float_t         pTuu_MindR_Sort4;
   std::vector<float>   *truth_pt;
   std::vector<float>   *truth_eta;
   std::vector<float>   *truth_phi;
   std::vector<float>   *truth_m;
   std::vector<int>     *truth_pdgid;
   std::vector<int>     *truth_status;
   std::vector<int>     *truth_barcode;
   std::vector<Long64_t> *truth_tthbb_info;
   std::vector<float>   *truth_jet_pt;
   std::vector<float>   *truth_jet_eta;
   std::vector<float>   *truth_jet_phi;
   std::vector<float>   *truth_jet_m;


   // Extra variables for output to the output TTree
   std::vector<float> ll_particle_px, ll_particle_py, ll_particle_pz, ll_particle_e, ll_particle_tagInfo;
   std::vector<int> ll_particle_type, ll_particle_recoInclusion, ll_particle_trueInclusion;
   int truth_decay_mode, lepton_count;
   int truth_decay_mode_old;
   float best_mWH_lvbb, best_mWH_qqbb, best_mH, best_mWqq, best_mWlv;
   Float_t         Lepton_Eta;
   Float_t         Lepton_Pt;
   Float_t         Lepton_Phi;
   Float_t         Lepton_M;
   Float_t         Truth_Higgs_Eta;
   Float_t         Truth_Higgs_Pt;
   Float_t         Truth_Higgs_Phi;
   Float_t         Truth_Higgs_M;
   Float_t         Truth_Wplus_Eta;
   Float_t         Truth_Wplus_Pt;
   Float_t         Truth_Wplus_Phi;
   Float_t         Truth_Wplus_M;
   Float_t         m_HT_bjets_Lepton_Pt;
   Float_t         m_pTH;
   Float_t         m_pTH_over_mVH_qqbb;
   Float_t         m_pTH_over_mVH_lvbb;
   Float_t         m_mH;
   Float_t         m_mass_resolution_qqbb;
   Float_t         m_mass_resolution_lvbb;
   Float_t         m_MET_over_sqrtHT;
   Float_t         m_pTW_leptonic;
   Float_t         m_mW_leptonic;
   Float_t         m_pTW_hadronic;
   Float_t         m_mW_hadronic;
   Float_t         m_LepEnergyFrac_qqbb;
   Float_t         m_LepEnergyFrac_lvbb;
   Float_t         m_deltaR_LH;
   Float_t         m_deltaR_LWhad;
   Float_t         m_deltaEta_HWhad;
   Float_t         m_deltaPhi_HWhad;
   Float_t         m_deltaEta_HWlep;
   Float_t         m_deltaPhi_HWlep;
   Float_t         ratio_Wpt_mVH_qqbb;
   Float_t         ratio_Wpt_mVH_lvbb;
   Float_t         ratio_Hpt_mVH_qqbb;
   Float_t         ratio_Hpt_mVH_lvbb;
   //Float_t         nJets;
   //Float_t         nFjets;


   // A map used for human-readable printout of truth information, for debugging
   std::unordered_map<int, std::string> pdgid_map = {
      { 1, "down quark" },
      { -1, "anti-down quark" },
      { 2, "up quark" },
      { -2, "anti-up quark" },
      { 3, "strange quark" },
      { -3, "anti-strange quark" },
      { 4, "charm quark" },
      { -4, "anti-charm quark" },
      { 5, "bottom quark" },
      { -5, "anti-bottom quark" },
      { 6, "top quark" },
      { -6, "anti-top quark" },
      { 11, "electron" },
      { -11, "positron" },
      { 12, "electron neutrino" },
      { -12, "anti-electron neutrino" },
      { 13, "muon" },
      { -13, "anti-muon" },
      { 14, "muon neutrino" },
      { -14, "anti-muon neutrino" },
      { 15, "tau" },
      { -15, "anti-tau" },
      { 16, "tau neutrino" },
      { -16, "anti-tau neutrino" },
      { 21, "gluon" },
      { 22, "photon" },
      { 23, "Z Boson" },
      { 24, "W+" },
      { -24, "W-" },
      { 25, "SM Higgs" },
      { 37, "+ve Charged Higgs" },
      { -37, "-ve Charged Higgs" }
   };


   // Useful vars for the low-level variable stuff:
   std::vector<Particle> particles;
   TLorentzVector ll_truth_Higgs, ll_truth_W, ll_truth_W_reco;
   TLorentzVector ll_truth_Higgs_old, ll_truth_W_old, ll_truth_W_reco_old;

   // Other vars:
   TString processName;
   std::map<TString, bool> pass_sel; // dict-like item which contains a bool determining if each of the different selection regions were passed.
   // SIGNAL_DSIDS = A vector of DSIDS corresponding to the files of the signal samples we're interested in.
   // This will be used to determine if we should try and use generator-level information or not (yes if DSID in this list, no if not)
   //std::vector<int> SIGNAL_DSIDS = {504558,504562,504565,504567,504569,504570,504571}; // Might be worth having this in a file and then read it up
   std::vector<int> SIGNAL_DSIDS = {510115,510116,510117,510118,510119,510120,510121,510122,510123,510124}; // Might be worth having this in a file and then read it up
   bool is_signal_sample; // A bool which will allow us to label if an MC sample is signal or not. For data this will always be false.
   uint dsid; // An int containing the dataset ID (DSID)
   int maxTreeSize; // Variable read from config to allow setting of the max tree size for output TTree file writing (so we don't write one giant file). If negative number, then we don't change from default (whcih I think is 100Gb)
   bool debugMode; // Variable read from config to allow debug mode, which runs over a small number of events but does masses of print out
   bool doCombined; // Variable read from config to allow Combined mode where the even/odd NN scores are combined into one branch
   bool WriteAllEvents; // Variable read from config to allow us to write all events (ie, not just those where the selection of 0, 3, 8, 9, 10 is passed)
   bool LowLevelDeltaRLepLjetCut; // Variable read from config to allow us to toggle whether or not the DeltaR(ljet, lep)>1.0 cut is on for the low-level particles
   bool lightWeightMode; // Variable read from config to allow lightweight mode, which only stores a small selection of our data
   std::string ttbarSelection; // Variable read from config to determine the ttbar weight selection mode. Can be one of: "Nominal", "Bfilt", "HTfilt", "Bfilt + HTfilt"
   bool lowlevel_output_mode = false;        // Variable read from config to allow write-out (to root TTree) of the low-level input variables (small-R jets and large-R jets, specifically). This should be turned off to save space if you don't need it
   bool category_cut_on;         // Variable read from config file to determine if a category cut should be applied (this refers to the different possible states of the 'selection_category' variable)
   bool category_0_enabled;      // Variable read from config file to detemine if 0 is an accepted category (only relevant when category_cut_on is true)
   bool category_1_enabled;      // Variable read from config file to detemine if 1 is an accepted category (only relevant when category_cut_on is true)
   bool category_2_enabled;      // Variable read from config file to detemine if 2 is an accepted category (only relevant when category_cut_on is true)
   bool category_3_enabled;      // Variable read from config file to detemine if 3 is an accepted category (only relevant when category_cut_on is true)
   bool category_4_enabled;      // Variable read from config file to detemine if 4 is an accepted category (only relevant when category_cut_on is true)
   bool category_5_enabled;      // Variable read from config file to detemine if 5 is an accepted category (only relevant when category_cut_on is true)
   bool category_6_enabled;      // Variable read from config file to detemine if 6 is an accepted category (only relevant when category_cut_on is true)
   bool category_7_enabled;      // Variable read from config file to detemine if 7 is an accepted category (only relevant when category_cut_on is true)
   std::vector<int> chosen_categories; // Vector to store the accepted categories (only revelant when category_cut_on is true)
   int min_n_fat_jets; 
   int max_n_fat_jets;
   // Now some bools to determine whether or not to apply a given cut
   // TODO not sure if these are in the right place; should they be inside the event loop?
   bool hmlb_cut_on; // Cut on SM Higgs mass lower bound on/off
   bool hmub_cut_on; // Cut on SM Higgs mass upper bound on/off
   bool wm_cut_on; // Cut on W boson mass on/off
   bool hw_angle_cut_on; // Cut on Minimum angle between the Higgs and Wboson on/off
   bool min_n_fat_jets_cut_on; // Cut on the minimum number of fat jets on/off
   bool max_n_fat_jets_cut_on; // Cut on the maximum number of fat jets on/off
   bool met_pt_min_cut_on; // Cut on Minimum missing transverse momentum on/off
   bool lep_pt_min_cut_on; // Cut on Minimum transverse momentum of lepton on/off
   bool higgs_pt_min_cut_on; // Cut on Minimum transverse momentum of Higgs jet on/off
   bool W_leptonic_pt_min_cut_on; // Cut on Minimum transverse momentum of W jet on/off
   bool lep_SMHiggs_angle_min_cut_on; // Cut on Minimum angle between lepton and Higgs jet on/off
   bool lep_SMHiggs_angle_max_cut_on; // Cut on Maximum angle between lepton and Higgs jet on/off
   bool lep_W_hadronic_angle_min_cut_on; // Cut on Maximum angle between lepton and W jet jet on/off
   bool status_W_cut_on; // Cut on W boson found or not (Will only affect leptonic decay mode)
   // Variables which were in the old ntuples, which we need to calculate using the new input variables using special functions at the start of the EventLoop Loop function.
   Float_t EventWeight; // This is what it was called in the old ntuples, and therefore in this code
   Int_t Lepton_Charge;
   TLorentzVector MET;
   TLorentzVector Higgs_Truth; // For storing the combination of truth b quarks that we assume to be the Higgs
   TLorentzVector Higgs_Truth_P; // For storing the actual TLorentzVector from the truth Higgs
   TLorentzVector Wplus_Truth; // For storing the combination of truth particles (quark pair or lep/neutrino pair) that we assume to be the Wplus
   TLorentzVector Wplus_Truth_P; // For storing the actual TLorentzVector from the truth Wplus
   std::vector<float> *FatJet_M;
   std::vector<float> *FatJet_PT;
   std::vector<float> *FatJet_Eta; // Unsure if used; may just be copied over but should include for now
   std::vector<float> *FatJet_Phi; // Unsure if used; may just be copied over but should include for now
   std::vector<float> *signal_Jet_M; // Used but maybe just for copying over (unsure at the moment); not sure what a 'signal jet' is; think this is just a particular choice of jet. 
   std::vector<float> *signal_Jet_E; // Used but maybe just for copying over (unsure at the moment); not sure what a 'signal jet' is; think this is just a particular choice of jet. 
   std::vector<float> *signal_Jet_PT; // Used but maybe just for copying over (unsure at the moment); not sure what a 'signal jet' is; think this is just a particular choice of jet. 
   std::vector<float> *signal_Jet_Eta; // Used but maybe just for copying over (unsure at the moment); not sure what a 'signal jet' is; think this is just a particular choice of jet. 
   std::vector<float> *signal_Jet_Phi; // Used but maybe just for copying over (unsure at the moment); not sure what a 'signal jet' is; think this is just a particular choice of jet. 
   //std::vector<float> *forward_Jet_M; // Used but maybe just for copying over (unsure at the moment); not sure what a 'forward jet' is; think this is just a particular choice of jet. 
   //std::vector<float> *forward_Jet_PT; // Used but maybe just for copying over (unsure at the moment); not sure what a 'forward jet' is; think this is just a particular choice of jet. 
   //std::vector<float> *forward_Jet_Eta; // Used but maybe just for copying over (unsure at the moment); not sure what a 'forward jet' is; think this is just a particular choice of jet. 
   //std::vector<float> *forward_Jet_Phi; // Used but maybe just for copying over (unsure at the moment); not sure what a 'forward jet' is; think this is just a particular choice of jet. 
   std::vector<float> *btag_score_signalJet; // Used but maybe just for copying over (unsure at the moment); not sure what a 'signal jet' is; think this is just a particular choice of jet. 
   std::vector<float> *btag_score_forwardJet; // Used but maybe just for copying over (unsure at the moment); not sure what a 'forward jet' is; think this is just a particular choice of jet. 
   std::vector<float> *TrackJet_PT; //---USED---
   std::vector<float> *TrackJet_Eta; //---USED---
   std::vector<float> *TrackJet_Phi; //---USED---
   std::vector<float> *TrackJet_M; //---USED---
   std::vector<float> *TrackJet_E; //---USED---
   std::vector<float> *TrackJet_btagWeight; //---USED---
   std::vector<int>   *signal_Jet_tagWeightBin_DL1r_Continuous;
   std::vector<int>   *track_Jet_tagWeightBin_DL1r_Continuous;
   TLorentzVector Higgs_p1;
   TLorentzVector Higgs_p2;
   TLorentzVector Wplus_p1;
   TLorentzVector Wplus_p2;
   bool     found_Higgs_constituents;
   bool     found_Wplus_had_constituents;
   bool     found_Wplus_lep_constituents;
   std::vector<char>    *el_LHTight;
   Float_t         weight_normalise; // Extra one copied from Simon's code
   Float_t         m_xsec; // The xsec for this process
   Float_t         m_kfac; // kfac for this process
   Float_t         m_sumOfMCGenWeights; // sum of Monte Carlo generated weights for this process
   float xbb_tag_higgsJet_value; // Will hold the value of the entry of b_ljet_Xbb2020v3_Higgs corresponding to the jet we choose to be the Higgs
   float Xbb_variable_FJet_Higgs; // Will hold the value of the variable calculated based upon the b_ljet_Xbb2020v3_Higgs corresponding to the jet we choose to be the Higgs
   float Xbb_variable_FJet_WHad; // If >1 Fat jet is present, will hold the value of the variable calculated based upon the b_ljet_Xbb2020v3_Higgs corresponding to the jet we choose to be the W
   int N_bInH; // Number of b tags found in the jet reconstructed H
   int N_bInW; // Number of b tags found in the hadronically reconstructed W
   int selection_category; // Category as defined in the final slide here: https://indico.cern.ch/event/1166094/contributions/4897394/attachments/2460706/4218843/Boosted_Update_13_06_2022.pdf
   int combined_category; // The final category in the final analysis. 0, 8, 10 map to 0 (lvbb) whilst 3, 9 map to 3 (qqbb)

   // List of branches
   TBranch        *b_mc_generator_weights;   //!
   TBranch        *b_weight_mc;   //!
   TBranch        *b_GenFiltHT;   //!
   TBranch        *b_weight_pileup;   //!
   TBranch        *b_weight_leptonSF;   //!
   TBranch        *b_weight_globalLeptonTriggerSF;   //!
   TBranch        *b_weight_oldTriggerSF;   //!
   TBranch        *b_weight_bTagSF_DL1r_Continuous;   //!
   TBranch        *b_weight_trackjet_bTagSF_DL1r_Continuous;   //!
   TBranch        *b_weight_jvt;   //!
   TBranch        *b_weight_pileup_UP;   //!
   TBranch        *b_weight_pileup_DOWN;   //!
   TBranch        *b_weight_leptonSF_EL_SF_Trigger_UP;   //!
   TBranch        *b_weight_leptonSF_EL_SF_Trigger_DOWN;   //!
   TBranch        *b_weight_leptonSF_EL_SF_Reco_UP;   //!
   TBranch        *b_weight_leptonSF_EL_SF_Reco_DOWN;   //!
   TBranch        *b_weight_leptonSF_EL_SF_ID_UP;   //!
   TBranch        *b_weight_leptonSF_EL_SF_ID_DOWN;   //!
   TBranch        *b_weight_leptonSF_EL_SF_Isol_UP;   //!
   TBranch        *b_weight_leptonSF_EL_SF_Isol_DOWN;   //!
   TBranch        *b_weight_leptonSF_MU_SF_Trigger_STAT_UP;   //!
   TBranch        *b_weight_leptonSF_MU_SF_Trigger_STAT_DOWN;   //!
   TBranch        *b_weight_leptonSF_MU_SF_Trigger_SYST_UP;   //!
   TBranch        *b_weight_leptonSF_MU_SF_Trigger_SYST_DOWN;   //!
   TBranch        *b_weight_leptonSF_MU_SF_ID_STAT_UP;   //!
   TBranch        *b_weight_leptonSF_MU_SF_ID_STAT_DOWN;   //!
   TBranch        *b_weight_leptonSF_MU_SF_ID_SYST_UP;   //!
   TBranch        *b_weight_leptonSF_MU_SF_ID_SYST_DOWN;   //!
   TBranch        *b_weight_leptonSF_MU_SF_ID_STAT_LOWPT_UP;   //!
   TBranch        *b_weight_leptonSF_MU_SF_ID_STAT_LOWPT_DOWN;   //!
   TBranch        *b_weight_leptonSF_MU_SF_ID_SYST_LOWPT_UP;   //!
   TBranch        *b_weight_leptonSF_MU_SF_ID_SYST_LOWPT_DOWN;   //!
   TBranch        *b_weight_leptonSF_MU_SF_Isol_STAT_UP;   //!
   TBranch        *b_weight_leptonSF_MU_SF_Isol_STAT_DOWN;   //!
   TBranch        *b_weight_leptonSF_MU_SF_Isol_SYST_UP;   //!
   TBranch        *b_weight_leptonSF_MU_SF_Isol_SYST_DOWN;   //!
   TBranch        *b_weight_leptonSF_MU_SF_TTVA_STAT_UP;   //!
   TBranch        *b_weight_leptonSF_MU_SF_TTVA_STAT_DOWN;   //!
   TBranch        *b_weight_leptonSF_MU_SF_TTVA_SYST_UP;   //!
   TBranch        *b_weight_leptonSF_MU_SF_TTVA_SYST_DOWN;   //!
   TBranch        *b_weight_globalLeptonTriggerSF_EL_Trigger_UP;   //!
   TBranch        *b_weight_globalLeptonTriggerSF_EL_Trigger_DOWN;   //!
   TBranch        *b_weight_globalLeptonTriggerSF_MU_Trigger_STAT_UP;   //!
   TBranch        *b_weight_globalLeptonTriggerSF_MU_Trigger_STAT_DOWN;   //!
   TBranch        *b_weight_globalLeptonTriggerSF_MU_Trigger_SYST_UP;   //!
   TBranch        *b_weight_globalLeptonTriggerSF_MU_Trigger_SYST_DOWN;   //!
   TBranch        *b_weight_oldTriggerSF_EL_Trigger_UP;   //!
   TBranch        *b_weight_oldTriggerSF_EL_Trigger_DOWN;   //!
   TBranch        *b_weight_oldTriggerSF_MU_Trigger_STAT_UP;   //!
   TBranch        *b_weight_oldTriggerSF_MU_Trigger_STAT_DOWN;   //!
   TBranch        *b_weight_oldTriggerSF_MU_Trigger_SYST_UP;   //!
   TBranch        *b_weight_oldTriggerSF_MU_Trigger_SYST_DOWN;   //!
   TBranch        *b_weight_jvt_UP;   //!
   TBranch        *b_weight_jvt_DOWN;   //!
   TBranch        *b_weight_bTagSF_DL1r_Continuous_eigenvars_B_up;   //!
   TBranch        *b_weight_bTagSF_DL1r_Continuous_eigenvars_C_up;   //!
   TBranch        *b_weight_bTagSF_DL1r_Continuous_eigenvars_Light_up;   //!
   TBranch        *b_weight_bTagSF_DL1r_Continuous_eigenvars_B_down;   //!
   TBranch        *b_weight_bTagSF_DL1r_Continuous_eigenvars_C_down;   //!
   TBranch        *b_weight_bTagSF_DL1r_Continuous_eigenvars_Light_down;   //!
   TBranch        *b_weight_trackjet_bTagSF_DL1r_Continuous_eigenvars_B_up;   //!
   TBranch        *b_weight_trackjet_bTagSF_DL1r_Continuous_eigenvars_C_up;   //!
   TBranch        *b_weight_trackjet_bTagSF_DL1r_Continuous_eigenvars_Light_up;   //!
   TBranch        *b_weight_trackjet_bTagSF_DL1r_Continuous_eigenvars_B_down;   //!
   TBranch        *b_weight_trackjet_bTagSF_DL1r_Continuous_eigenvars_C_down;   //!
   TBranch        *b_weight_trackjet_bTagSF_DL1r_Continuous_eigenvars_Light_down;   //!
   TBranch        *b_eventNumber;   //!
   TBranch        *b_runNumber;   //!
   TBranch        *b_randomRunNumber;   //!
   TBranch        *b_mcChannelNumber;   //!
   TBranch        *b_mu;   //!
   TBranch        *b_mu_actual;   //!
   TBranch        *b_el_pt;   //!
   TBranch        *b_el_eta;   //!
   TBranch        *b_el_cl_eta;   //!
   TBranch        *b_el_phi;   //!
   TBranch        *b_el_e;   //!
   TBranch        *b_el_charge;   //!
   TBranch        *b_el_topoetcone20;   //!
   TBranch        *b_el_ptvarcone20;   //!
   TBranch        *b_el_isTight;   //!
   TBranch        *b_el_CF;   //!
   TBranch        *b_el_d0sig;   //!
   TBranch        *b_el_delta_z0_sintheta;   //!
   TBranch        *b_mu_pt;   //!
   TBranch        *b_mu_eta;   //!
   TBranch        *b_mu_phi;   //!
   TBranch        *b_mu_e;   //!
   TBranch        *b_mu_charge;   //!
   TBranch        *b_mu_topoetcone20;   //!
   TBranch        *b_mu_ptvarcone30;   //!
   TBranch        *b_mu_isTight;   //!
   TBranch        *b_mu_d0sig;   //!
   TBranch        *b_mu_delta_z0_sintheta;   //!
   TBranch        *b_jet_pt;   //!
   TBranch        *b_jet_eta;   //!
   TBranch        *b_jet_phi;   //!
   TBranch        *b_jet_e;   //!
   TBranch        *b_jet_jvt;   //!
   TBranch        *b_jet_truthflav;   //!
   TBranch        *b_jet_truthflavExtended;   //!
   TBranch        *b_jet_isbtagged_DL1r_60;   //!
   TBranch        *b_jet_isbtagged_DL1r_70;   //!
   TBranch        *b_jet_isbtagged_DL1r_77;   //!
   TBranch        *b_jet_isbtagged_DL1r_85;   //!
   TBranch        *b_jet_tagWeightBin_DL1r_Continuous;   //!
   TBranch        *b_jet_DL1r;   //!
   TBranch        *b_ljet_pt;   //!
   TBranch        *b_ljet_eta;   //!
   TBranch        *b_ljet_phi;   //!
   TBranch        *b_ljet_e;   //!
   TBranch        *b_ljet_m;   //!
   TBranch        *b_ljet_truthLabel;   //!
   TBranch        *b_tjet_pt;   //!
   TBranch        *b_tjet_eta;   //!
   TBranch        *b_tjet_phi;   //!
   TBranch        *b_tjet_e;   //!
   TBranch        *b_tjet_tagWeightBin_DL1r_Continuous;   //!
   TBranch        *b_tjet_DL1r;   //!
   TBranch        *b_met_met;   //!
   TBranch        *b_met_phi;   //!
   TBranch        *b_ejets_2018_DL1r;   //!
   TBranch        *b_mujets_2018_DL1r;   //!
   TBranch        *b_boosted_ljets_ejets_2018_DL1r;   //!
   TBranch        *b_boosted_ljets_mujets_2018_DL1r;   //!
   TBranch        *b_HLT_mu26_ivarmedium;   //!
   TBranch        *b_HLT_mu50;   //!
   TBranch        *b_HLT_e60_lhmedium_nod0;   //!
   TBranch        *b_HLT_e140_lhloose_nod0;   //!
   TBranch        *b_HLT_e26_lhtight_nod0_ivarloose;   //!
   TBranch        *b_ljet_C2;   //!
   TBranch        *b_ljet_D2;   //!
   TBranch        *b_ljet_Xbb2020v3_Higgs;   //!
   TBranch        *b_ljet_Xbb2020v3_QCD;   //!
   TBranch        *b_ljet_Xbb2020v3_Top;   //!
   TBranch        *b_ljet_muonCorrectedEta;   //!
   TBranch        *b_ljet_muonCorrectedMass;   //!
   TBranch        *b_ljet_muonCorrectedPhi;   //!
   TBranch        *b_ljet_muonCorrectedPt;   //!
   TBranch        *b_HF_Classification;   //!
   TBranch        *b_HF_ClassificationGhost;   //!
   TBranch        *b_HF_SimpleClassification;   //!
   TBranch        *b_HF_SimpleClassificationGhost;   //!
   TBranch        *b_TopHeavyFlavorFilterFlag;   //!
   TBranch        *b_nBTagsTrackJets_DL1r_60;   //!
   TBranch        *b_nBTagsTrackJets_DL1r_70;   //!
   TBranch        *b_nBTagsTrackJets_DL1r_77;   //!
   TBranch        *b_nBTagsTrackJets_DL1r_85;   //!
   TBranch        *b_nBTags_DL1r_60;   //!
   TBranch        *b_nBTags_DL1r_70;   //!
   TBranch        *b_nBTags_DL1r_77;   //!
   TBranch        *b_nBTags_DL1r_85;   //!
   TBranch        *b_nElectrons;   //!
   TBranch        *b_nJets;   //!
   TBranch        *b_nLJets;   //!
   TBranch        *b_nLJets_matched;   //!
   TBranch        *b_nMuons;   //!
   TBranch        *b_nPDFFlavor;   //!
   TBranch        *b_nPrimaryVtx;   //!
   TBranch        *b_nTaus;   //!
   TBranch        *b_Aplanarity_bjets_77;   //!
   TBranch        *b_Aplanarity_bjets_85;   //!
   TBranch        *b_Aplanarity_bjets_Sort4;   //!
   TBranch        *b_Aplanarity_jets;   //!
   TBranch        *b_Centrality_all;   //!
   TBranch        *b_H0_all;   //!
   TBranch        *b_H1_all;   //!
   TBranch        *b_H2_jets;   //!
   TBranch        *b_H4_all;   //!
   TBranch        *b_HT_all;   //!
   TBranch        *b_HT_jets;   //!
   TBranch        *b_Mbb_HiggsMass_77;   //!
   TBranch        *b_Mbb_HiggsMass_85;   //!
   TBranch        *b_Mbb_HiggsMass_Sort4;   //!
   TBranch        *b_Mbb_MaxM_77;   //!
   TBranch        *b_Mbb_MaxM_85;   //!
   TBranch        *b_Mbb_MaxM_Sort4;   //!
   TBranch        *b_Mbb_MaxPt_77;   //!
   TBranch        *b_Mbb_MaxPt_85;   //!
   TBranch        *b_Mbb_MaxPt_Sort4;   //!
   TBranch        *b_Mbb_MinM_77;   //!
   TBranch        *b_Mbb_MinM_85;   //!
   TBranch        *b_Mbb_MinM_Sort4;   //!
   TBranch        *b_Mbb_MindR_77;   //!
   TBranch        *b_Mbb_MindR_85;   //!
   TBranch        *b_Mbb_MindR_Sort4;   //!
   TBranch        *b_Mbj_MaxPt_77;   //!
   TBranch        *b_Mbj_MaxPt_85;   //!
   TBranch        *b_Mbj_MaxPt_Sort4;   //!
   TBranch        *b_Mbj_MindR_77;   //!
   TBranch        *b_Mbj_MindR_85;   //!
   TBranch        *b_Mbj_MindR_Sort4;   //!
   TBranch        *b_Mbj_Wmass_77;   //!
   TBranch        *b_Mbj_Wmass_85;   //!
   TBranch        *b_Mbj_Wmass_Sort4;   //!
   TBranch        *b_Mjj_HiggsMass;   //!
   TBranch        *b_Mjj_MaxPt;   //!
   TBranch        *b_Mjj_MinM;   //!
   TBranch        *b_Mjj_MindR;   //!
   TBranch        *b_Mjjj_MaxPt;   //!
   TBranch        *b_Muu_MindR_77;   //!
   TBranch        *b_Muu_MindR_85;   //!
   TBranch        *b_Muu_MindR_Sort4;   //!
   TBranch        *b_dRbb_MaxM_77;   //!
   TBranch        *b_dRbb_MaxM_85;   //!
   TBranch        *b_dRbb_MaxM_Sort4;   //!
   TBranch        *b_dRbb_MaxPt_77;   //!
   TBranch        *b_dRbb_MaxPt_85;   //!
   TBranch        *b_dRbb_MaxPt_Sort4;   //!
   TBranch        *b_dRbb_MindR_77;   //!
   TBranch        *b_dRbb_MindR_85;   //!
   TBranch        *b_dRbb_MindR_Sort4;   //!
   TBranch        *b_dRbb_avg_77;   //!
   TBranch        *b_dRbb_avg_85;   //!
   TBranch        *b_dRbb_avg_Sort4;   //!
   TBranch        *b_dRbj_Wmass_77;   //!
   TBranch        *b_dRbj_Wmass_85;   //!
   TBranch        *b_dRbj_Wmass_Sort4;   //!
   TBranch        *b_dRlepbb_MindR_77;   //!
   TBranch        *b_dRlepbb_MindR_85;   //!
   TBranch        *b_dRlepbb_MindR_Sort4;   //!
   TBranch        *b_dRlj_MindR;   //!
   TBranch        *b_dRuu_MindR_77;   //!
   TBranch        *b_dRuu_MindR_85;   //!
   TBranch        *b_dRuu_MindR_Sort4;   //!
   TBranch        *b_pT_jet3;   //!
   TBranch        *b_pT_jet5;   //!
   TBranch        *b_pTbb_MindR_77;   //!
   TBranch        *b_pTbb_MindR_85;   //!
   TBranch        *b_pTbb_MindR_Sort4;   //!
   TBranch        *b_pTuu_MindR_77;   //!
   TBranch        *b_pTuu_MindR_85;   //!
   TBranch        *b_pTuu_MindR_Sort4;   //!
   TBranch        *b_truth_pt;   //!
   TBranch        *b_truth_eta;   //!
   TBranch        *b_truth_phi;   //!
   TBranch        *b_truth_m;   //!
   TBranch        *b_truth_pdgid;   //!
   TBranch        *b_truth_status;   //!
   TBranch        *b_truth_barcode;   //!
   TBranch        *b_truth_tthbb_info;   //!
   TBranch        *b_truth_jet_pt;   //!
   TBranch        *b_truth_jet_eta;   //!
   TBranch        *b_truth_jet_phi;   //!
   TBranch        *b_truth_jet_m;   //!

   EventLoop(TTree *tree = 0, TString sampleName = "", TString ExpUncertaintyName = "Nominal", TString WP = "",
             bool UnweightedCutFlow = true, bool WeightedCutFlow = true, bool UnweightedAltCutFlow = false, bool WeightedAltCutFlow = false,
             Float_t hmlb = 90., Float_t hmub = 140., Float_t wmlb = 70., Float_t wmub = 100., Float_t met_pt_min = 30000.,
             Float_t lep_pt_min = 30000., Float_t higgs_pt_min = 200000., Float_t W_leptonic_pt_min = 200000., Float_t lep_SMHiggs_angle_min = 1.0, Float_t lep_SMHiggs_angle_max = 1.0,
             Float_t lep_W_hadronic_angle_min = 1.0, Float_t hw_angle = 2.5);
   EventLoop(TTree *tree, TString ExpUncertaintyName, TString outFileName, std::unordered_map<std::string, std::string> config);
   void Write(TDirectory *dir, std::string dirname);
   void WriteTreeToFile(TFile *outFile);
   void SetDebugMode(bool debugMode);
   void FillMVATree(int i_H1, int i_H2, int i_w1, int i_w2, bool is_signal); // TODO Pretty sure this is unused for the boosted channel (currently) but is needed to ensure that we don't error because it's called in MatchTruthParticlesToJets
   void Sort_Jets(std::vector<TLorentzVector> *Jets, std::vector<int> *is_tagged);
   void Sort_vec1_by_vec2(std::vector<float> *vec1, std::vector<float> *vec2);
   void ApplySmallRjetRemoval(std::vector<TLorentzVector> *smallRJets, std::vector<int> *smallRJetsIsTagged, TLorentzVector largeRJet);
   void Sort_Jets_and_vars(std::vector<TLorentzVector> *Jets, std::vector<int> *is_tagged, std::vector<float> *vec1, std::vector<float> *vec2, std::vector<float> *vec3);
   // TODO see if we want to use the Set_Jet_observables function; it seems to calculate some useful info for us; resolved search used it but Zahaab didn't for some reason. Adds at least the DeltaPhiJETMET and the m_HT_bjets getting output
   void CalcEventWeight();
   void btagCounting();
   void Set_Jet_observables(); // unused but possibly want to use this?, was for the resolved jets search
   void SetTruthParticles();
   void SetJetVectors();
   bool SetLeptonVectors();
   void FindTop();
   // void WriteMVAInput(); // unused, was for the resolved jets search
   void MatchTruthParticlesToJets();
   //void initializeMVA_qqbb(); // unused, was for the resolved jets search
   //void initializeMVA_lvbb(); // unused, was for the resolved jets search
   double GetMwt();
   double GetTruthMass();
   //double EvaluateMVAResponse_qqbb(int i_H1, int i_H2, int i_w1, int i_w2); // unused, was for the resolved jets search
   //double EvaluateMVAResponse_lvbb(int i_H1, int i_H2, TLorentzVector W); // unused, was for the resolved jets search
   //bool FindJetPair_qqbb(); // unused, was for the resolved jets search
   //bool FindJetPair_lvbb(); // unused, was for the resolved jets search
   bool FindFJetPair(Float_t higgs_pt_min, Float_t W_leptonic_pt_min, 
                     Float_t lep_SMHiggs_angle_min, Float_t lep_SMHiggs_angle_max, 
                     Float_t lep_W_hadronic_angle_min, Float_t hw_angle);

   void SetJetPair();
   TLorentzVector GetWFromJets();
   TLorentzVector GetHFromJets();
   //bool PassEventSelectionResolved(); // unused, was for the resolved jets search
   bool PassEventSelectionBoosted(Float_t met_pt_min, Float_t lep_pt_min, 
                                    Float_t higgs_pt_min, Float_t W_leptonic_pt_min, 
                                    Float_t lep_SMHiggs_angle_min, Float_t lep_SMHiggs_angle_max, 
                                    Float_t lep_W_hadronic_angle_min, Float_t hw_angle);
   int GetBTagCategory(int NTags_InHiggsJet, int NTags_OutsideHiggsJet);
   int GetBTagCategoryShort(int NTags_InHiggsJet, int NTags_OutsideHiggsJet);
   int GetTagWeightBin(double btag_score);
   TLorentzVector GetWBoson(bool &status);
   float SetNormFactor(float xsec, float kfac, float sumOfMCGenWeights);
   float SetLumiFactor(float luminosity_factor);
   //TLorentzVector BuildLeptonicTop(); // unused
   std::vector<TLorentzVector *> GetNeutrinos(TLorentzVector *L, TLorentzVector *MET);
   bool TranslateVariables();
   void ResetVariables();
   bool WriteEventOutStdout();
   bool WriteEventOutHist();
   bool WriteEventOutCsv();
   virtual ~EventLoop();
   virtual Int_t    Cut(Long64_t entry);
   virtual Int_t    GetEntry(Long64_t entry);
   virtual Long64_t LoadTree(Long64_t entry);
   virtual void     Init(TTree *tree, TString sampleName, TString ExpUncertaintyName);
   virtual void     Loop();
   virtual Bool_t   Notify();
   virtual void     Show(Long64_t entry = -1);
   virtual bool     LowLevel_Loop();
   virtual int      LowLevel_CountLeptons();
   virtual int      LowLevel_ClassifyDecayType();
   virtual int      LowLevel_ClassifyDecayType_OLD();
   virtual void LowLevel_MatchTruthParticles();
   virtual std::tuple<float, float, float> LowLevel_GetBestWhMasses();
   virtual void     Fill_NN_Scores();
   // Chi2_minimization *myMinimizer = new Chi2_minimization("MeV"); // unused
   NeutrinoBuilder *m_NeutrinoBuilder;

   
   std::vector<double> m_EventWeights;
   std::vector<TString> m_UncNames;
   std::vector<TString> mySel;
   
   std::vector<TLorentzVector> bQuarks;
   std::vector<TLorentzVector> LightQuarks;   
   std::vector<TLorentzVector> Leptons;
   std::vector<TLorentzVector> GenLevLeptons;
   std::vector<int> GenLevLeptonsPdgid;
   std::vector<TLorentzVector> Jets;
   std::vector<float> Jets_Pt; // all small-R jets
   std::vector<float> Jets_Eta; // all small-R jets
   std::vector<float> Jets_Phi; // all small-R jets
   std::vector<float> Jets_M; // all small-R jets
   std::vector<float> Jets_tagWeightBinDL1rContinuous; // all small-R jets
   std::vector<TLorentzVector> FJets; // TODO should rename this to lJets or something, to match current naming convention
   std::vector<float> FJets_Pt; // b-tagged large-R jets
   std::vector<float> FJets_Eta; // b-tagged large-R jets
   std::vector<float> FJets_Phi; // b-tagged large-R jets
   std::vector<float> FJets_M; // b-tagged large-R jets
   std::vector<float> FJets_DXbb; // b-tagged large-R jets
   std::vector<float> ljet_Xbb2020v3_Higgs_REARRANGED;
   std::vector<float> ljet_Xbb2020v3_QCD_REARRANGED;
   std::vector<float> ljet_Xbb2020v3_Top_REARRANGED;
   std::vector<TLorentzVector> TrkJets;
   std::vector<int> JetIsTaggedBin;
   //std::vector<int> JetIsTagged; // TODO don't think this is used anymore
   std::vector<int> Jets_PCbtag;
   std::vector<int> TrkJets_PCbtag;
   std::vector<int> TrkJetIsTaggedBin; // Also unused (as TrkJets)
   //std::vector<int> nTaggedVRTrkJetsInFJet; // No longer used, requirement replaced by btag counting function

   // TODO Work out which of the below values (down to the line '//tests') are used and unused (only looked at the first few)
   TLorentzVector Top_lepto; // This will be a top reconstructed using the leptonic W and the MET
   TLorentzVector Top_hadro; // This will be a top reconstructed using the hadronic W and the MET
   TLorentzVector Higgs;
   TLorentzVector W_leptonic;
   TLorentzVector W_hadronic;
   TLorentzVector W_hadronic_FromSmallRJets; // Allows us to store information from both cases when we're not sure how to treat it
   TLorentzVector W_hadronic_FromLargeRJet; // Allows us to store information from both cases when we're not sure how to treat it
   int index_W_jet_1; // For if we reconstruct the W (from H+) hadronically using small-R jets, index of the first jet which makes this up?
   int index_W_jet_2; // For if we reconstruct the W (from H+) hadronically using small-R jets, index of the first jet which makes this up?
   int index_H_jet_1; // For if we reconstruct the W (from H+) hadronically using small-R jets, index of the first jet which makes this up?
   int index_H_jet_2; // For if we reconstruct the W (from H+) hadronically using small-R jets, index of the first jet which makes this up?
   //TLorentzVector Wplus;
   int m_is_Signal;
   int m_NTags;
   int m_NTags_caloJ; // TODO What is the difference between caloJ and trkJ in our code?
   int m_NTags_trkJ; // TODO Seems unused, set to zero then written out without changing. Not commenting out yet as it requires other things to be commented out

   int m_diff_trk_calo_btags;
   int m_diff_trk_calo_jets;

   // Variables we will create for writing out to our output TTree/Histograms/Csv/stdout/ wherever we want to write out to
   int m_NTags_Higgs; // TODO Seems unused, set to zero then written out without changing. Not commenting out yet as it requires other things to be commented out
   int m_NTags_Wplus; // TODO Seems unused, set to zero then nothing else done
   int m_ntagsOutside; // TODO Seems unused, set to zero then written out without changing. Not commenting out yet as it requires other things to be commented out
   int m_nTrkTagsOutside;
   int m_nTrkTagsInW;
   int m_nTrkTagsInH;
   int m_nTrkTagsOutside_smallR;
   int m_nTrkTagsInW_smallR;
   int m_nTrkTagsInH_smallR;
   int m_nTrkTagsOutside_largeR;
   int m_nTrkTagsInW_largeR;
   int m_nTrkTagsInH_largeR;
   int m_bTagCategory;
   int m_btagCategoryBin;// TODO Seems unused, set to a value based upon WP then nothing else done
   int m_index_H1;
   int m_index_H2;
   int m_index_W1;
   int m_index_W2;
   double m_min_dRTruth;
   double m_min_chi2;
   double m_min_DeltaPhiJETMET;
   double m_DeltaPhi_H_Lep;
   double m_DeltaPhi_H_MET;
   double m_DeltaPhi_W_hadronic_Lep;
   double m_DeltaPhi_W_hadronic_MET;
   double m_MaxMVA_Response;
   double m_HT;
   double m_HT_bjets;
   double m_maxEta_bjets;
   double m_maxPT_bjets;
   double m_MET;
   double m_mWT;
   double m_Lep_PT;
   double m_mVH_qqbb;
   double m_mVH_lvbb;
   double m_mVH_qqbb_WFromSmallRJets;
   double m_mVH_qqbb_WFromLargeRJet;
   double m_DeltaPhi_HW_hadronic;
   double m_DeltaPhi_HW_leptonic;
   double m_DeltaR_HW_hadronic;
   double m_DeltaR_HW_leptonic;
   double m_Wleptonic_pT;
   double m_Wleptonic_Eta;
   double m_MassTruth;
   //float m_H_mass;
   float m_H_pT;
   float m_pTjH1;
   float m_pTjH2;
   float m_btagjH1;
   float m_btagjH2;
   float m_dRjjH;
   float m_Wp_mass;
   float m_Wp_pT;
   float m_pTjWp1;
   float m_pTjWp2;
   float m_btagjWp1;
   float m_btagjWp2;
   float m_dRjjWp;
   float m_Phi_HW;
   float m_mass_VH;
   float m_pTH_over_mvH_qqbb;
   float m_pTH_over_mvH_lvbb;
   float m_ptW_over_mvH_qqbb;
   float m_ptW_over_mvH_lvbb;
   float m_mTop_lepto;
   float m_mTop_hadro;
   float m_mTop_best;
   float m_H_mass;
   float m_H_phi;
   float m_H_eta;
   float m_H_Pt;
   float m_Whad_mass;
   float m_Whad_phi;
   float m_Whad_eta;
   float m_Whad_Pt;
   float m_Whad_FromSmallRJets_mass;
   float m_Whad_FromSmallRJets_phi;
   float m_Whad_FromSmallRJets_eta;
   float m_Whad_FromSmallRJets_Pt;
   float m_Whad_FromLargeRJet_mass;
   float m_Whad_FromLargeRJet_phi;
   float m_Whad_FromLargeRJet_eta;
   float m_Whad_FromLargeRJet_Pt;
   float m_Wlep_mass;
   float m_Wlep_phi;
   float m_Wlep_eta;
   float m_Wlep_Pt;
   float m_lep_mass;
   float m_lep_phi;
   float m_lep_eta;
   float m_lep_Pt;
   //tests
   //Int_t bins; // unused
   //Double_t integral; // unused
   // Higgs and W mass bounds
   Float_t hmlb = 90.;
   Float_t hmub = 140.;
   Float_t wmlb = 70.;
   Float_t wmub = 100.;
   // Variables for cuts. Key: 
   // Label used for it's value,      Name of value in code,                              What it's called in physics
   // met_pt_min,                   MET->pt(),                                      missing transverse momentum
   // lep_pt_min,                   Lepton4vector->Pt(),                            lepton transverse momentum
   // higgs_pt_min,                  FJets.at(0)->pt(),                              transverse momenta of fat bjet 0
   // lep_SMHiggs_angle_min,            FJets.at(0).DeltaR(*Lepton4vector),             mninimum angle of fat bjet 0 and lepton
   // lep_SMHiggs_angle_max,        
   // hw_angle,                  m_DeltaPhi_HW,                                  angle of higgs and wboson (minimum accepted for cut)
   // I refer to the jet that is asigned to FJets.at(0) as jet 0 and the FJets.at(1) jet as jet 1
   Float_t met_pt_min = 30000.;
   Float_t lep_pt_min = 30000.;
   Float_t higgs_pt_min = 200000.;
   Float_t W_leptonic_pt_min = 200000.;
   Float_t lep_SMHiggs_angle_min = 0.0;
   Float_t lep_SMHiggs_angle_max = 1.0;
   Float_t lep_W_hadronic_angle_min = 1.0;
   Float_t hw_angle = 2.5;
   // TODO work out whether Weighted cut flow or Unweighted cut flow is the one we want, similarly for Alt vs not
   bool weights_included = true; // This will be yes for Monte Carlo, no for data
   bool is_ttbar = false; // This determines if we need to do flavour filtering or not
   float luminosity_factor = -1.0;
   std::string truth_W_decay_mode = "None"; // This will be set depending on generator level info; "leptonic" means lv, "hadronic" means qq
   int truth_lep_charge = 0; // Also set depending on generator level info; +1/-1 means truth event, 0 means no truth available
   int truth_agreement = 0; // Also set depending on generator level info; +1/-1 means two methods agree/disagree, 0 means at least one was not set
   int lep_charge_agreement = 0; // Also set depending on generator level info; +1/-1 means truth/reco lepton charges agree/disagree, 0 means at least one was not set
   bool UnweightedCutFlow = true;
   bool WeightedCutFlow = true;
   bool UnweightedAltCutFlow = false;
   bool WeightedAltCutFlow = false;

   //DeltaR for small jets removal
   double Small_Jet_removal = 1.4;

   /////////////////////////////////////////////
   // Some stuff for NN predictions
   /////////////////////////////////////////////
   std::map<std::string, std::string> nn_name_map_lvbb = {
      // // {"NN_lvbb_Final_EvenTrained","/Users/sidbaines/Documents/PhD/Work/HplusWh/MLEventSel/nn_cpp_preds/20230427.HpWhNNs/lvbbNN_trainedOnEven.json"},
      // // {"NN_lvbb_Final_OddTrained","/Users/sidbaines/Documents/PhD/Work/HplusWh/MLEventSel/nn_cpp_preds/20230427.HpWhNNs/lvbbNN_trainedOnOdd.json"},
      // // {"NN_lvbb_Old_OddTrained","/Users/sidbaines/Documents/PhD/Work/HplusWh/MLEventSel/nn_cpp_preds/lvbb_NNv2_gt1.25TeV_flatAbswtdTrain__128relul22e-05_128relul22e-05_128relul22e-05__lr1e-05_bs128_p12_tf_fold0_FULLlwtnn.json"},  
      // // 
      // // 
      {"NN_lvbb_Final_OddTrained","/users/baines/Code/ChargedHiggs_CodeForThesis/lwtnnModels/Final_NNs/lvbbNN_trainedOnOdd.json"},
      {"NN_lvbb_Final_EvenTrained","/users/baines/Code/ChargedHiggs_CodeForThesis/lwtnnModels/Final_NNs/lvbbNN_trainedOnEven.json"},
      // //
      // //
      // {"jNN_nomTtbar_lvbb_OddTrained","/Users/sidbaines/Documents/PhD/Work/HplusWh/MLEventSel_OnlyImportant/lwtnns/DifferentTtbarInclusionTrainings3/lvbb_NNv1_gt0.95TeV_MCEqualCappedBkgRwtdCmbDownwtdTrain__64relul20.0001_64relul20.0001_drop0.05__lr1e-05_bs128_p10_combBkg_tf_ttbar_nom_fold0_FULLlwtnn.json"},
      // {"jNN_nomTtbar_lvbb_EvenTrained","/Users/sidbaines/Documents/PhD/Work/HplusWh/MLEventSel_OnlyImportant/lwtnns/DifferentTtbarInclusionTrainings3/lvbb_NNv1_gt0.95TeV_MCEqualCappedBkgRwtdCmbDownwtdTrain__64relul20.0001_64relul20.0001_drop0.05__lr1e-05_bs128_p10_combBkg_tf_ttbar_nom_fold1_FULLlwtnn.json"},
      // {"jNN_nomPlusFlavTtbar_lvbb_OddTrained","/Users/sidbaines/Documents/PhD/Work/HplusWh/MLEventSel_OnlyImportant/lwtnns/DifferentTtbarInclusionTrainings3/lvbb_NNv3_gt0.95TeV_MCEqualCappedBkgRwtdCmbDownwtdTrain__64relul20.0001_64relul20.0001_drop0.05__lr1e-05_bs128_p10_combBkg_tf_ttbar_nomPlusB_fold0_FULLlwtnn.json"},
      // {"jNN_nomPlusFlavTtbar_lvbb_EvenTrained","/Users/sidbaines/Documents/PhD/Work/HplusWh/MLEventSel_OnlyImportant/lwtnns/DifferentTtbarInclusionTrainings3/lvbb_NNv3_gt0.95TeV_MCEqualCappedBkgRwtdCmbDownwtdTrain__64relul20.0001_64relul20.0001_drop0.05__lr1e-05_bs128_p10_combBkg_tf_ttbar_nomPlusB_fold1_FULLlwtnn.json"},
      // {"jNN_nomPlusHtTtbar_lvbb_OddTrained","/Users/sidbaines/Documents/PhD/Work/HplusWh/MLEventSel_OnlyImportant/lwtnns/DifferentTtbarInclusionTrainings3/lvbb_NNv5_gt0.95TeV_MCEqualCappedBkgRwtdCmbDownwtdTrain__64relul20.0001_64relul20.0001_drop0.05__lr1e-05_bs128_p10_combBkg_tf_ttbar_nomPlusHT_fold0_FULLlwtnn.json"},
      // {"jNN_nomPlusHtTtbar_lvbb_EvenTrained","/Users/sidbaines/Documents/PhD/Work/HplusWh/MLEventSel_OnlyImportant/lwtnns/DifferentTtbarInclusionTrainings3/lvbb_NNv5_gt0.95TeV_MCEqualCappedBkgRwtdCmbDownwtdTrain__64relul20.0001_64relul20.0001_drop0.05__lr1e-05_bs128_p10_combBkg_tf_ttbar_nomPlusHT_fold1_FULLlwtnn.json"},
      // {"jNN_nomPlusFlavPlusHtTtbar_lvbb_OddTrained","/Users/sidbaines/Documents/PhD/Work/HplusWh/MLEventSel_OnlyImportant/lwtnns/DifferentTtbarInclusionTrainings3/lvbb_NNv7_gt0.95TeV_MCEqualCappedBkgRwtdCmbDownwtdTrain__64relul20.0001_64relul20.0001_drop0.05__lr1e-05_bs128_p10_combBkg_tf_ttbar_nomPlusBPlusHT_fold0_FULLlwtnn.json"},
      // {"jNN_nomPlusFlavPlusHtTtbar_lvbb_EvenTrained","/Users/sidbaines/Documents/PhD/Work/HplusWh/MLEventSel_OnlyImportant/lwtnns/DifferentTtbarInclusionTrainings3/lvbb_NNv7_gt0.95TeV_MCEqualCappedBkgRwtdCmbDownwtdTrain__64relul20.0001_64relul20.0001_drop0.05__lr1e-05_bs128_p10_combBkg_tf_ttbar_nomPlusBPlusHT_fold1_FULLlwtnn.json"},

      // lvbb pNN vs jNN vs iNN
      // {"jNN_lvbb_OddTrained","/Users/sidbaines/Documents/PhD/Work/HplusWh/MLEventSel_OnlyImportant/lwtnns/lvbb_PNN_vs_JNN_vsINN/lvbb_jNN_trainedOnOdd_FULLlwtnn.json"},
      // {"pNN800_lvbb_OddTrained","/Users/sidbaines/Documents/PhD/Work/HplusWh/MLEventSel_OnlyImportant/lwtnns/lvbb_PNN_vs_JNN_vsINN/lvbb_pNN800_trainedOnOdd_FULLlwtnn.json"},
      // {"pNN900_lvbb_OddTrained","/Users/sidbaines/Documents/PhD/Work/HplusWh/MLEventSel_OnlyImportant/lwtnns/lvbb_PNN_vs_JNN_vsINN/lvbb_pNN900_trainedOnOdd_FULLlwtnn.json"},
      // {"pNN1000_lvbb_OddTrained","/Users/sidbaines/Documents/PhD/Work/HplusWh/MLEventSel_OnlyImportant/lwtnns/lvbb_PNN_vs_JNN_vsINN/lvbb_pNN1000_trainedOnOdd_FULLlwtnn.json"},
      // {"pNN1200_lvbb_OddTrained","/Users/sidbaines/Documents/PhD/Work/HplusWh/MLEventSel_OnlyImportant/lwtnns/lvbb_PNN_vs_JNN_vsINN/lvbb_pNN1200_trainedOnOdd_FULLlwtnn.json"},
      // {"pNN1400_lvbb_OddTrained","/Users/sidbaines/Documents/PhD/Work/HplusWh/MLEventSel_OnlyImportant/lwtnns/lvbb_PNN_vs_JNN_vsINN/lvbb_pNN1400_trainedOnOdd_FULLlwtnn.json"},
      // {"pNN1600_lvbb_OddTrained","/Users/sidbaines/Documents/PhD/Work/HplusWh/MLEventSel_OnlyImportant/lwtnns/lvbb_PNN_vs_JNN_vsINN/lvbb_pNN1600_trainedOnOdd_FULLlwtnn.json"},
      // {"pNN1800_lvbb_OddTrained","/Users/sidbaines/Documents/PhD/Work/HplusWh/MLEventSel_OnlyImportant/lwtnns/lvbb_PNN_vs_JNN_vsINN/lvbb_pNN1800_trainedOnOdd_FULLlwtnn.json"},
      // {"pNN2000_lvbb_OddTrained","/Users/sidbaines/Documents/PhD/Work/HplusWh/MLEventSel_OnlyImportant/lwtnns/lvbb_PNN_vs_JNN_vsINN/lvbb_pNN2000_trainedOnOdd_FULLlwtnn.json"},
      // {"pNN2500_lvbb_OddTrained","/Users/sidbaines/Documents/PhD/Work/HplusWh/MLEventSel_OnlyImportant/lwtnns/lvbb_PNN_vs_JNN_vsINN/lvbb_pNN2500_trainedOnOdd_FULLlwtnn.json"},  
      // {"pNN3000_lvbb_OddTrained","/Users/sidbaines/Documents/PhD/Work/HplusWh/MLEventSel_OnlyImportant/lwtnns/lvbb_PNN_vs_JNN_vsINN/lvbb_pNN3000_trainedOnOdd_FULLlwtnn.json"},  
      // {"iNN800_lvbb_OddTrained","/Users/sidbaines/Documents/PhD/Work/HplusWh/MLEventSel_OnlyImportant/lwtnns/lvbb_PNN_vs_JNN_vsINN/lvbb_iNN800_trainedOnOdd_FULLlwtnn.json"},
      // {"iNN900_lvbb_OddTrained","/Users/sidbaines/Documents/PhD/Work/HplusWh/MLEventSel_OnlyImportant/lwtnns/lvbb_PNN_vs_JNN_vsINN/lvbb_iNN900_trainedOnOdd_FULLlwtnn.json"},
      // {"iNN1000_lvbb_OddTrained","/Users/sidbaines/Documents/PhD/Work/HplusWh/MLEventSel_OnlyImportant/lwtnns/lvbb_PNN_vs_JNN_vsINN/lvbb_iNN1000_trainedOnOdd_FULLlwtnn.json"},
      // {"iNN1200_lvbb_OddTrained","/Users/sidbaines/Documents/PhD/Work/HplusWh/MLEventSel_OnlyImportant/lwtnns/lvbb_PNN_vs_JNN_vsINN/lvbb_iNN1200_trainedOnOdd_FULLlwtnn.json"},
      // {"iNN1400_lvbb_OddTrained","/Users/sidbaines/Documents/PhD/Work/HplusWh/MLEventSel_OnlyImportant/lwtnns/lvbb_PNN_vs_JNN_vsINN/lvbb_iNN1400_trainedOnOdd_FULLlwtnn.json"},
      // {"iNN1600_lvbb_OddTrained","/Users/sidbaines/Documents/PhD/Work/HplusWh/MLEventSel_OnlyImportant/lwtnns/lvbb_PNN_vs_JNN_vsINN/lvbb_iNN1600_trainedOnOdd_FULLlwtnn.json"},
      // {"iNN1800_lvbb_OddTrained","/Users/sidbaines/Documents/PhD/Work/HplusWh/MLEventSel_OnlyImportant/lwtnns/lvbb_PNN_vs_JNN_vsINN/lvbb_iNN1800_trainedOnOdd_FULLlwtnn.json"},
      // {"iNN2000_lvbb_OddTrained","/Users/sidbaines/Documents/PhD/Work/HplusWh/MLEventSel_OnlyImportant/lwtnns/lvbb_PNN_vs_JNN_vsINN/lvbb_iNN2000_trainedOnOdd_FULLlwtnn.json"},
      // {"iNN2500_lvbb_OddTrained","/Users/sidbaines/Documents/PhD/Work/HplusWh/MLEventSel_OnlyImportant/lwtnns/lvbb_PNN_vs_JNN_vsINN/lvbb_iNN2500_trainedOnOdd_FULLlwtnn.json"},
      // {"iNN3000_lvbb_OddTrained","/Users/sidbaines/Documents/PhD/Work/HplusWh/MLEventSel_OnlyImportant/lwtnns/lvbb_PNN_vs_JNN_vsINN/lvbb_iNN3000_trainedOnOdd_FULLlwtnn.json"},
   };
   std::map<std::string, std::string> nn_name_map_qqbb = {
      {"NN_qqbb_Final_OddTrained","/users/baines/Code/ChargedHiggs_CodeForThesis/lwtnnModels/Final_NNs/qqbbNN_trainedOnOdd.json"},
      {"NN_qqbb_Final_EvenTrained","/users/baines/Code/ChargedHiggs_CodeForThesis/lwtnnModels/Final_NNs/qqbbNN_trainedOnEven.json"},
      // qqbb pNN vs iNN vs jNN
      {"jNN_qqbb_OddTrained","/users/baines/Code/ChargedHiggs_CodeForThesis/lwtnnModels/Old/qqbb_NNv2_gt0.95TeV_MCEqualCappedBkgRwtdCmbDownwtdTrain__64relul20.0001_64relul20.0001_drop0.05__lr3e-05_bs128_p10_combBkg_tf_ttbar_nom_fold0_FULLlwtnn.json"},
      {"pNN800_qqbb_OddTrained","/users/baines/Code/ChargedHiggs_CodeForThesis/lwtnnModels/Old/qqbb_pNNv1_gt0.95TeV_MCEqualCappedBkgRwtdCmbDownwtdTrain__64relul20.0001_64relul20.0001_drop0.05__lr3e-05_bs128_p10_combBkg_tf_ttbar_nom_fold0_FULLlwtnn.json"},
      {"pNN900_qqbb_OddTrained","/users/baines/Code/ChargedHiggs_CodeForThesis/lwtnnModels/Old/qqbb_pNNv1_gt0.95TeV_MCEqualCappedBkgRwtdCmbDownwtdTrain__64relul20.0001_64relul20.0001_drop0.05__lr3e-05_bs128_p10_combBkg_tf_ttbar_nom_fold0_FULLlwtnn.json"},
      {"pNN1000_qqbb_OddTrained","/users/baines/Code/ChargedHiggs_CodeForThesis/lwtnnModels/Old/qqbb_pNNv1_gt0.95TeV_MCEqualCappedBkgRwtdCmbDownwtdTrain__64relul20.0001_64relul20.0001_drop0.05__lr3e-05_bs128_p10_combBkg_tf_ttbar_nom_fold0_FULLlwtnn.json"},
      {"pNN1200_qqbb_OddTrained","/users/baines/Code/ChargedHiggs_CodeForThesis/lwtnnModels/Old/qqbb_pNNv1_gt0.95TeV_MCEqualCappedBkgRwtdCmbDownwtdTrain__64relul20.0001_64relul20.0001_drop0.05__lr3e-05_bs128_p10_combBkg_tf_ttbar_nom_fold0_FULLlwtnn.json"},
      {"pNN1400_qqbb_OddTrained","/users/baines/Code/ChargedHiggs_CodeForThesis/lwtnnModels/Old/qqbb_pNNv1_gt0.95TeV_MCEqualCappedBkgRwtdCmbDownwtdTrain__64relul20.0001_64relul20.0001_drop0.05__lr3e-05_bs128_p10_combBkg_tf_ttbar_nom_fold0_FULLlwtnn.json"},
      {"pNN1600_qqbb_OddTrained","/users/baines/Code/ChargedHiggs_CodeForThesis/lwtnnModels/Old/qqbb_pNNv1_gt0.95TeV_MCEqualCappedBkgRwtdCmbDownwtdTrain__64relul20.0001_64relul20.0001_drop0.05__lr3e-05_bs128_p10_combBkg_tf_ttbar_nom_fold0_FULLlwtnn.json"},
      {"pNN1800_qqbb_OddTrained","/users/baines/Code/ChargedHiggs_CodeForThesis/lwtnnModels/Old/qqbb_pNNv1_gt0.95TeV_MCEqualCappedBkgRwtdCmbDownwtdTrain__64relul20.0001_64relul20.0001_drop0.05__lr3e-05_bs128_p10_combBkg_tf_ttbar_nom_fold0_FULLlwtnn.json"},
      {"pNN2000_qqbb_OddTrained","/users/baines/Code/ChargedHiggs_CodeForThesis/lwtnnModels/Old/qqbb_pNNv1_gt0.95TeV_MCEqualCappedBkgRwtdCmbDownwtdTrain__64relul20.0001_64relul20.0001_drop0.05__lr3e-05_bs128_p10_combBkg_tf_ttbar_nom_fold0_FULLlwtnn.json"},
      {"pNN2500_qqbb_OddTrained","/users/baines/Code/ChargedHiggs_CodeForThesis/lwtnnModels/Old/qqbb_pNNv1_gt0.95TeV_MCEqualCappedBkgRwtdCmbDownwtdTrain__64relul20.0001_64relul20.0001_drop0.05__lr3e-05_bs128_p10_combBkg_tf_ttbar_nom_fold0_FULLlwtnn.json"},
      {"pNN3000_qqbb_OddTrained","/users/baines/Code/ChargedHiggs_CodeForThesis/lwtnnModels/Old/qqbb_pNNv1_gt0.95TeV_MCEqualCappedBkgRwtdCmbDownwtdTrain__64relul20.0001_64relul20.0001_drop0.05__lr3e-05_bs128_p10_combBkg_tf_ttbar_nom_fold0_FULLlwtnn.json"},
      {"iNN800_qqbb_OddTrained","/users/baines/Code/ChargedHiggs_CodeForThesis/lwtnnModels/Old/qqbb_NNv3_gt0.95TeV_MCEqualCappedBkgRwtdCmbDownwtdTrain__64relul20.0001_64relul20.0001_drop0.05__lr3e-05_bs128_p10_combBkg_tf_ttbar_nom_fold0_FULLlwtnn.json"},
      {"iNN900_qqbb_OddTrained","/users/baines/Code/ChargedHiggs_CodeForThesis/lwtnnModels/Old/qqbb_NNv4_gt0.95TeV_MCEqualCappedBkgRwtdCmbDownwtdTrain__64relul20.0001_64relul20.0001_drop0.05__lr3e-05_bs128_p10_combBkg_tf_ttbar_nom_fold0_FULLlwtnn.json"},
      {"iNN1000_qqbb_OddTrained","/users/baines/Code/ChargedHiggs_CodeForThesis/lwtnnModels/Old/qqbb_NNv5_gt0.95TeV_MCEqualCappedBkgRwtdCmbDownwtdTrain__64relul20.0001_64relul20.0001_drop0.05__lr3e-05_bs128_p10_combBkg_tf_ttbar_nom_fold0_FULLlwtnn.json"},
      {"iNN1200_qqbb_OddTrained","/users/baines/Code/ChargedHiggs_CodeForThesis/lwtnnModels/Old/qqbb_NNv6_gt0.95TeV_MCEqualCappedBkgRwtdCmbDownwtdTrain__64relul20.0001_64relul20.0001_drop0.05__lr3e-05_bs128_p10_combBkg_tf_ttbar_nom_fold0_FULLlwtnn.json"},
      {"iNN1400_qqbb_OddTrained","/users/baines/Code/ChargedHiggs_CodeForThesis/lwtnnModels/Old/qqbb_NNv7_gt0.95TeV_MCEqualCappedBkgRwtdCmbDownwtdTrain__64relul20.0001_64relul20.0001_drop0.05__lr3e-05_bs128_p10_combBkg_tf_ttbar_nom_fold0_FULLlwtnn.json"},
      {"iNN1600_qqbb_OddTrained","/users/baines/Code/ChargedHiggs_CodeForThesis/lwtnnModels/Old/qqbb_NNv8_gt0.95TeV_MCEqualCappedBkgRwtdCmbDownwtdTrain__64relul20.0001_64relul20.0001_drop0.05__lr3e-05_bs128_p10_combBkg_tf_ttbar_nom_fold0_FULLlwtnn.json"},
      {"iNN1800_qqbb_OddTrained","/users/baines/Code/ChargedHiggs_CodeForThesis/lwtnnModels/Old/qqbb_NNv9_gt0.95TeV_MCEqualCappedBkgRwtdCmbDownwtdTrain__64relul20.0001_64relul20.0001_drop0.05__lr3e-05_bs128_p10_combBkg_tf_ttbar_nom_fold0_FULLlwtnn.json"},
      {"iNN2000_qqbb_OddTrained","/users/baines/Code/ChargedHiggs_CodeForThesis/lwtnnModels/Old/qqbb_NNv10_gt0.95TeV_MCEqualCappedBkgRwtdCmbDownwtdTrain__64relul20.0001_64relul20.0001_drop0.05__lr3e-05_bs128_p10_combBkg_tf_ttbar_nom_fold0_FULLlwtnn.json"},
      {"iNN2500_qqbb_OddTrained","/users/baines/Code/ChargedHiggs_CodeForThesis/lwtnnModels/Old/qqbb_NNv11_gt0.95TeV_MCEqualCappedBkgRwtdCmbDownwtdTrain__64relul20.0001_64relul20.0001_drop0.05__lr3e-05_bs128_p10_combBkg_tf_ttbar_nom_fold0_FULLlwtnn.json"},
      {"iNN3000_qqbb_OddTrained","/users/baines/Code/ChargedHiggs_CodeForThesis/lwtnnModels/Old/qqbb_NNv12_gt0.95TeV_MCEqualCappedBkgRwtdCmbDownwtdTrain__64relul20.0001_64relul20.0001_drop0.05__lr3e-05_bs128_p10_combBkg_tf_ttbar_nom_fold0_FULLlwtnn.json"},
   };
   std::map<std::string, lwt::LightweightGraph*> nn_map_lvbb, nn_map_qqbb; // Maps of the output branch names to the actual lwtnn NN objects
   std::ifstream in_file_nn;
   lwt::GraphConfig nn_config;
   // Inputs for the NN need to be in a std::map
   // First, the lvbb
   std::map<std::string, double> m_inputs_lvbb = {
      {"LepEnergyFrac_lvbb", 0.0},
      {"mTop_lepto", 0.0},
      {"dR_LH", 0.0},
      {"DeltaEta_H_WLep", 0.0},
      {"RWpTM_lvbb", 0.0},
      {"RHpTM_lvbb", 0.0},
      {"DeltaPhi_H_WLep", 0.0}
   };
   // Now, the qqbb
   std::map<std::string, double> m_inputs_qqbb = {
      {"LepEnergyFrac_qqbb", 0.0},
      {"RWpTM_qqbb", 0.0},
      {"RHpTM_qqbb", 0.0},
      {"dR_LWhad", 0.0},
      {"dR_LH", 0.0},
      {"DeltaPhi_H_WHad", 0.0},
      {"DeltaEta_H_WHad", 0.0}
   };
   // Make more output tree branches
   // Create and connect the prediction variables/branches
   std::map<std::string, double> lvbbNNs, qqbbNNs; // Maps of the output branch names to the places where the predicitons are stored, so we can connect them
   std::string NN_basename, NN_foldname;
   std::size_t basename_delim_pos;
   std::map<std::string, std::map<std::string, double>> mm_inputs;
   std::map<std::string, double> outputs;
   /////////////////////////////////////////////
   // END OF Some stuff for NN predictions
   /////////////////////////////////////////////

   //cutflow of config parameters
   std::ofstream m_cutFlowFileStream;
   std::ofstream m_cutFlowParametersFileStream;
   std::ofstream m_cutFlowFileStreamAlt;
   TString m_cutFlowFileName;
   TString m_cutFlowParametersFileName;
   TString m_cutFlowFileNameAlt;
   struct CutFlowType
   {
      std::unordered_map<std::string, std::unordered_map<std::string, int> > unweighted_vals =
      {
         {"both_channels",
            {
               {"zeroTags", 0},
               {"oneTags", 0},
               {"twoTags", 0},
               {"threeTags", 0},
               {"fourPlusTags", 0}
            }
         },
         {"jjbb_channel",
            {
               {"zeroTags", 0},
               {"oneTags", 0},
               {"twoTags", 0},
               {"threeTags", 0},
               {"fourPlusTags", 0}
            }
         },
         {"lvbb_channel",
            {
               {"zeroTags", 0},
               {"oneTags", 0},
               {"twoTags", 0},
               {"threeTags", 0},
               {"fourPlusTags", 0}
            }
         }
      };

      std::unordered_map<std::string, std::unordered_map<std::string, Float_t> > weighted_vals = 
      {
         {"both_channels",
            {
               {"zeroTags", 0},
               {"oneTags", 0},
               {"twoTags", 0},
               {"threeTags", 0},
               {"fourPlusTags", 0}
            }
         },
         {"jjbb_channel",
            {
               {"zeroTags", 0},
               {"oneTags", 0},
               {"twoTags", 0},
               {"threeTags", 0},
               {"fourPlusTags", 0}
            }
         },
         {"lvbb_channel",
            {
               {"zeroTags", 0},
               {"oneTags", 0},
               {"twoTags", 0},
               {"threeTags", 0},
               {"fourPlusTags", 0}
            }
         }
      };

      void increment_unweighted(std::string channel, int ntags)
      {
         //assert(("ntags input to CutFlowType::increment_unweighted must be >=1", ntags > 0));
         if (ntags < 1) unweighted_vals[channel]["zeroTags"]++; // TODO Should this (and elsewhere) be '==0' instead, to reserve negative values for errors?
         if (ntags == 1) unweighted_vals[channel]["oneTags"]++;
         if (ntags == 2) unweighted_vals[channel]["twoTags"]++;
         if (ntags == 3) unweighted_vals[channel]["threeTags"]++;
         if (ntags > 3) unweighted_vals[channel]["fourPlusTags"]++;
      }

      void increment_weighted(std::string channel, int ntags, Float_t weight)
      {
         //assert(("ntags input to CutFlowType::increment_unweighted must be >=1", ntags > 0));
         if (ntags < 1) weighted_vals[channel]["zeroTags"] = weighted_vals[channel]["zeroTags"] + weight;
         if (ntags == 1) weighted_vals[channel]["oneTags"] = weighted_vals[channel]["oneTags"] + weight;
         if (ntags == 2) weighted_vals[channel]["twoTags"] = weighted_vals[channel]["twoTags"] + weight;
         if (ntags == 3) weighted_vals[channel]["threeTags"] = weighted_vals[channel]["threeTags"] + weight;
         if (ntags > 3) weighted_vals[channel]["fourPlusTags"] = weighted_vals[channel]["fourPlusTags"] + weight;
      }

      int jjbb_UnweightedTotal()
      {
         return unweighted_vals["jjbb_channel"]["zeroTags"] + unweighted_vals["jjbb_channel"]["oneTags"] + unweighted_vals["jjbb_channel"]["twoTags"] + unweighted_vals["jjbb_channel"]["threeTags"] + unweighted_vals["jjbb_channel"]["fourPlusTags"];
      }

      Float_t jjbb_WeightedTotal()
      {
         return weighted_vals["jjbb_channel"]["zeroTags"] + weighted_vals["jjbb_channel"]["oneTags"] + weighted_vals["jjbb_channel"]["twoTags"] + weighted_vals["jjbb_channel"]["threeTags"] + weighted_vals["jjbb_channel"]["fourPlusTags"];
      }

      int lvbb_UnweightedTotal()
      {
         return unweighted_vals["lvbb_channel"]["zeroTags"] + unweighted_vals["lvbb_channel"]["oneTags"] + unweighted_vals["lvbb_channel"]["twoTags"] + unweighted_vals["lvbb_channel"]["threeTags"] + unweighted_vals["lvbb_channel"]["fourPlusTags"];
      }

      Float_t lvbb_WeightedTotal()
      {
         return weighted_vals["lvbb_channel"]["zeroTags"] + weighted_vals["lvbb_channel"]["oneTags"] + weighted_vals["lvbb_channel"]["twoTags"] + weighted_vals["lvbb_channel"]["threeTags"] + weighted_vals["lvbb_channel"]["fourPlusTags"];
      }
      
      int both_UnweightedTotal()
      {
         return unweighted_vals["both_channels"]["zeroTags"] + unweighted_vals["both_channels"]["oneTags"] + unweighted_vals["both_channels"]["twoTags"] + unweighted_vals["both_channels"]["threeTags"] + unweighted_vals["both_channels"]["fourPlusTags"];
      }

      Float_t both_WeightedTotal()
      {
         return weighted_vals["both_channels"]["zeroTags"] + weighted_vals["both_channels"]["oneTags"] + weighted_vals["both_channels"]["twoTags"] + weighted_vals["both_channels"]["threeTags"] + weighted_vals["both_channels"]["fourPlusTags"];
      }
      
   };

   void altCutFlow(Float_t met_pt_min, Float_t lep_pt_min, Float_t higgs_pt_min, 
                     Float_t W_leptonic_pt_min, Float_t lep_SMHiggs_angle_min, 
                     Float_t lep_SMHiggs_angle_max, Float_t lep_W_hadronic_angle_min, Float_t hw_angle);
   void CutFlowAssignment(CutFlowType &cutVariable, bool XUnweightedCutFlow, bool XWeightedCutFlow);
   void CutFlowParser(std::ofstream &File, CutFlowType &cutVariable, const std::string cutName, bool XUnweightedCutFlow, bool XWeightedCutFlow);
   CutFlowType m_TotalEvents;
   CutFlowType m_noJets;
   //CutFlowType m_insufficientTags;
   CutFlowType m_MinNFatJetsCutFlow;
   CutFlowType m_MaxNFatJetsCutFlow;
   CutFlowType m_METCutFlow;
   CutFlowType m_LeptonPtCutFlow;
   CutFlowType m_WPtCutFlow;
   CutFlowType m_selectionCategoryCutFlow;
   CutFlowType m_ChannelFlexCutFlow;
   CutFlowType m_HiggsPtCutFlow;
   CutFlowType m_Higgs_LeptonAngleCutflow;
   CutFlowType m_Higgs_MaxLeptonAngleCutflow;
   CutFlowType m_W_hadronic_LeptonAngleCutflow;
   CutFlowType m_Higgs_WplusAngleCutflow;
   CutFlowType m_PositiveLepWCutflow;
   //CutFlowType m_PositiveLepHiggsPtCutFlow; No longer handled separately from the Two Fat Jet case
   CutFlowType m_HiggsMassCutFlow;
   CutFlowType m_WplusMassCutFlow;
};

#endif

#ifdef EventLoop_cxx
EventLoop::EventLoop(TTree *tree, TString ExpUncertaintyName, TString outFileName, std::unordered_map<std::string, std::string> config) : fChain(tree) 
{
// if parameter tree is not specified (or zero), fail
   if (tree == 0) {
      std::cerr << "Error in EventLoop::EventLoop(): tree is nullptr" << std::endl;
      return;
   }


   if (config["ttbarSelection"] != "")
   {
      ttbarSelection = config["ttbarSelection"];
   }
   if (config["Debug_Mode"] != "")
   {
      debugMode = (config["Debug_Mode"] == "Enable" || config["Debug_Mode"] == "enable");
   }
   if (config["maxTreeSize"] != "")
   {
      maxTreeSize = std::stoi(config["maxTreeSize"]);
   }
   if (config["do_combined_NN"] != "")
   {
      doCombined = (config["do_combined_NN"] == "Enable" || config["do_combined_NN"] == "enable");
   }
   if (config["write_all_events"] != "")
   {
      WriteAllEvents = (config["write_all_events"] == "Enable" || config["write_all_events"] == "enable");
   }
   if (config["low_level_delta_R_lep_ljet_cut"] != "")
   {
      LowLevelDeltaRLepLjetCut = (config["low_level_delta_R_lep_ljet_cut"] == "Enable" || config["low_level_delta_R_lep_ljet_cut"] == "enable");
   }
   if (config["lightWeightMode"] != "")
   {
      lightWeightMode = (config["lightWeightMode"] == "Enable" || config["lightWeightMode"] == "enable");
   }
   if (config["Low_Level_output"] != "")
   {
      lowlevel_output_mode = (config["Low_Level_output"] == "Enable" || config["Low_Level_output"] == "enable");
   }
   if (config["activate_category_cut"] != "")
   {
      category_cut_on = (config["activate_category_cut"] == "Enable" || config["activate_category_cut"] == "enable");
      if (debugMode) std::cout << "\t" << "category_cut_on: " << category_cut_on << std::endl;
   }
   if (debugMode) std::cout << "\t\t" << "First loop over chosen categories" << std::endl;
   for(int cat : chosen_categories){
      if (debugMode) std::cout << "\t\t" << "Chosen categories includes: " << cat << std::endl;
   }
   if (config["category_0_enabled"] == "Enable" || config["category_0_enabled"] == "enable") chosen_categories.push_back(0);
   if (config["category_1_enabled"] == "Enable" || config["category_1_enabled"] == "enable") chosen_categories.push_back(1);
   if (config["category_2_enabled"] == "Enable" || config["category_2_enabled"] == "enable") chosen_categories.push_back(2);
   if (config["category_3_enabled"] == "Enable" || config["category_3_enabled"] == "enable") chosen_categories.push_back(3);
   if (config["category_4_enabled"] == "Enable" || config["category_4_enabled"] == "enable") chosen_categories.push_back(4);
   if (config["category_5_enabled"] == "Enable" || config["category_5_enabled"] == "enable") chosen_categories.push_back(5);
   if (config["category_6_enabled"] == "Enable" || config["category_6_enabled"] == "enable") chosen_categories.push_back(6);
   if (config["category_7_enabled"] == "Enable" || config["category_7_enabled"] == "enable") chosen_categories.push_back(7);
   if (config["category_8_enabled"] == "Enable" || config["category_8_enabled"] == "enable") chosen_categories.push_back(8);
   if (config["category_9_enabled"] == "Enable" || config["category_9_enabled"] == "enable") chosen_categories.push_back(9);
   if (config["category_10_enabled"] == "Enable" || config["category_10_enabled"] == "enable") chosen_categories.push_back(10);
   if (debugMode) std::cout << "\t\t" << "Second loop over chosen categories" << std::endl;
   for(int cat : chosen_categories){
      if (debugMode) std::cout << "\t\t" << "Chosen categories includes: " << cat << std::endl;
   }
   if (config["Min_number_fat_jets"] != "")
   {
      min_n_fat_jets = std::stoi(config["Min_number_fat_jets"]);
      assert(("min_n_fat_jets must be == 1 or 2", min_n_fat_jets == 1 || min_n_fat_jets == 2));
   }
   if (config["Max_number_fat_jets"] != "")
   {
      max_n_fat_jets = std::stoi(config["Max_number_fat_jets"]);
      assert(("max_n_fat_jets must be >= max_n_fat_jets", max_n_fat_jets >= min_n_fat_jets));
   }
   // Determine which cuts are active
   // TODO put in an error message (so it's clear why/how/when it fails) for things being missing
   // from the config structure/file
   if (config["activate_higgs_mass_lower_bound_cut"] != "")
   {
      hmlb_cut_on = true; // Higgs mass lower bound cut always on, at a default of 90GeV.
   }
   hmlb_cut_on = true; // Higgs mass lower bound cut always on, at a default of 90GeV.
   if (config["activate_higgs_mass_upper_bound_cut"] != "")
   {
      hmub_cut_on = true; // Higgs mass upper bound cut always on, at a default of 140GeV.
   }
   hmub_cut_on = true; // Higgs mass upper bound cut always on, at a default of 140GeV.
   if (config["activate_Wboson_mass_cut"] != "")
   {
      wm_cut_on = config["activate_WBoson_mass_cut"] == "Enable";
   }
   if (config["activate_Higgs_Wboson_angle_cut"] != "")
   {
      hw_angle_cut_on = config["activate_Higgs_Wboson_angle_cut"] == "Enable";
   }
   if (config["activate_Min_n_fat_jets_cut"] != "")
   {
      min_n_fat_jets_cut_on = config["activate_Min_n_fat_jets_cut"] == "Enable";
   }
   if (config["activate_Max_n_fat_jets_cut"] != "")
   {
      max_n_fat_jets_cut_on = config["activate_Max_n_fat_jets_cut"] == "Enable";
   }
   if (config["activate_Missing_transverse_momentum_cut"] != "")
   {
      met_pt_min_cut_on = true; // MET cut always on, at a default for 30GeV.
   }
   met_pt_min_cut_on = true; // MET cut always on, at a default for 30GeV.
   if (config["activate_Lepton_transverse_momentum_cut"] != "")
   {
      lep_pt_min_cut_on = config["activate_Lepton_transverse_momentum_cut"] == "Enable";
   }
   if (config["activate_2Jets_Jet1_transverse_momentum_cut"] != "")
   {
      higgs_pt_min_cut_on = config["activate_2Jets_Jet1_transverse_momentum_cut"]  == "Enable";
   }
   if (config["activate_W_leptonic_pt_min_cut"] != "")
   {
      W_leptonic_pt_min_cut_on = config["activate_W_leptonic_pt_min_cut"]  == "Enable";
   }
   if (config["activate_lepton_Higgs_angle_MIN_cut"] != "")
   {
      lep_SMHiggs_angle_min_cut_on = config["activate_lepton_Higgs_angle_MIN_cut"] == "Enable";
   }
   if (config["activate_lepton_Higgs_angle_MAX_cut"] != "")
   {
      lep_SMHiggs_angle_max_cut_on = config["activate_lepton_Higgs_angle_MAX_cut"] == "Enable";
   }
   if (config["activate_lepton_W_hadronic_angle_cut"] != "")
   {
      lep_W_hadronic_angle_min_cut_on = config["activate_lepton_W_hadronic_angle_cut"] == "Enable";
   }
   if (config["activate_StatusW_cut"] != "")
   {
      status_W_cut_on = config["activate_StatusW_cut"] == "Enable";
   }



   // Get the cut variables
   if (config["Higgs_mass_lower_bound"] != "")
   {
      hmlb = std::stof(config["Higgs_mass_lower_bound"]);
   }
   if (config["Higgs_mass_upper_bound"] != "")
   {
      hmub = std::stof(config["Higgs_mass_upper_bound"]);
   }
   if (config["Wboson_mass_lower_bound"] != "")
   {
      wmlb = std::stof(config["Wboson_mass_lower_bound"]);
   }
   if (config["Wboson_mass_upper_bound"] != "")
   {
      wmub = std::stof(config["Wboson_mass_upper_bound"]);
   }
   if (config["Higgs_Wboson_angle"] != "")
   {
      hw_angle = std::stof(config["Higgs_Wboson_angle"]);
   }
   if (config["Missing_transverse_momentum"] != "")
   {
      met_pt_min = std::stof(config["Missing_transverse_momentum"]);
   }
   if (config["Lepton_transverse_momentum"] != "")
   {
      lep_pt_min = std::stof(config["Lepton_transverse_momentum"]);
   }
   if (config["2Jets_Jet1_transverse_momentum"] != "")
   {
      higgs_pt_min = std::stof(config["2Jets_Jet1_transverse_momentum"]);
   }
   if (config["W_leptonic_pt_min"] != "")
   {
      W_leptonic_pt_min = std::stof(config["W_leptonic_pt_min"]);
   }
   if (config["2Jets_Jet1_lepton_angle"] != "")
   {
      lep_SMHiggs_angle_min = std::stof(config["2Jets_Jet1_lepton_angle"]);
   }
   if (config["2Jets_Jet1_lepton_angle_MAX"] != "")
   {
      lep_SMHiggs_angle_max = std::stof(config["2Jets_Jet1_lepton_angle_MAX"]);
   }
   if (config["2Jets_Jet2_lepton_angle"] != "")
   {
      lep_W_hadronic_angle_min = std::stof(config["2Jets_Jet2_lepton_angle"]);
   }
   if (config["Has_weights"] != "") // TODO Rewrite this (and three similar clauses below) to be more succinct. DONE!
   {
      weights_included = (config["Has_weights"] == "Enable" || config["Has_weights"] == "enable");
   }
   if (config["Is_Ttbar"] != "")
   {
      is_ttbar = (config["Is_Ttbar"] == "Enable" || config["Is_Ttbar"] == "enable");
   }
   if (config["UnweightedCutFlow"] != "") // TODO Rewrite this (and three similar clauses below) to be more succinct. DONE!
   {
      UnweightedCutFlow = (config["UnweightedCutFlow"] == "Enable" || config["UnweightedCutFlow"] == "enable");
   }
   if (config["WeightedCutFlow"] != "")
   {
      WeightedCutFlow = (config["WeightedCutFlow"] == "Enable" || config["WeightedCutFlow"] == "enable");
   }
   if (config["UnweightedAlternateCutFlow"] != "")
   {
      UnweightedAltCutFlow = (config["UnweightedAlternateCutFlow"] == "Enable" || config["UnweightedAlternateCutFlow"] == "enable");
   }
   if (config["WeightedAlternateCutFlow"] != "")
   {
      WeightedAltCutFlow = (config["WeightedAlternateCutFlow"] == "Enable" || config["WeightedAlternateCutFlow"] == "enable");
   }
   if (config["SampleName"] == "")
   {
      throw std::runtime_error(std::string("Error: No SampleName given"));
   }
   if (config["FileName"] == "")
   {
      throw std::runtime_error(std::string("Error: No FileName given"));
   }
   if (config["WP"] == "")
   {
      throw std::runtime_error(std::string("Error: No WP given"));
   }

   TString SampleName = config["SampleName"];
   TString FileName = config["FileName"];
   TString WP = config["WP"];

   //m_btagCut_value_trkJets = -1.; // Why is this line here?
   m_btagCategoryBin = 10; // If not set then  don't have a well defined working point. Value of 10 means that we fail ALL Btags
   // TODO What happens if the working point is none of the below? Should there be a fail?
   if (WP == "85p")
   {
      //m_btagCut_value_trkJets = 0.05;
      //m_btagCut_value_CaloJets = 0.665;
      m_btagCategoryBin = 2;
   }
   if (WP == "77p")
   {
      //m_btagCut_value_trkJets = 0.58;
      //m_btagCategoryBin_trkJets = 3; // to replace above line. TODO is this right??
      //m_btagCut_value_CaloJets = 0.64;
      m_btagCategoryBin = 3;
   }
   if (WP == "70p")
   {
      //m_btagCut_value_trkJets = 0.79;
      //m_btagCategoryBin_trkJets = 4; // to replace above line. TODO is this right??
      //m_btagCut_value_CaloJets = 0.83;
      //m_btagCategoryBin = 3;
      m_btagCategoryBin = 4;
   }
   if (WP == "60p")
   {
      //m_btagCut_value_trkJets = 0.92;
      //m_btagCategoryBin_trkJets = 5; // to replace above line. TODO is this right??
      //m_btagCut_value_CaloJets = 0.94;
      //m_btagCategoryBin = 4;
      m_btagCategoryBin = 5;
   }
   //std::cout << "outFileName 1 : " << outFileName << std::endl;
   outFileName.ReplaceAll(".root", "-cutFlow.csv");
   m_cutFlowFileName = outFileName;
   //std::cout << "outFileName 2 : " << outFileName << std::endl;
   outFileName.ReplaceAll("-cutFlow.csv", "-cutFlowAlt.csv");
   m_cutFlowFileNameAlt = outFileName;
   //std::cout << "outFileName 3 : " << outFileName << std::endl;
   outFileName.ReplaceAll("-cutFlowAlt.csv", "-cutFlowParameters.csv");
   m_cutFlowParametersFileName = outFileName;

   outFileName.ReplaceAll("-cutFlowParameters.csv", "-Machine_Learning_output.csv");

   // Sort out some stuff for NN
   for (auto & kv_pair:nn_name_map_lvbb){
      in_file_nn.open(kv_pair.second);
      nn_config = lwt::parse_json_graph(in_file_nn);
      nn_map_lvbb[kv_pair.first] = new lwt::LightweightGraph(nn_config);
      in_file_nn.close();
      in_file_nn.clear();
   }
   for (auto & kv_pair:nn_name_map_qqbb){
      in_file_nn.open(kv_pair.second);
      nn_config = lwt::parse_json_graph(in_file_nn);
      nn_map_qqbb[kv_pair.first] = new lwt::LightweightGraph(nn_config);
      in_file_nn.close();
      in_file_nn.clear();
   }

   // Create the new, output TTree and define it's branches. These will just be the variables we are interested in for whatever follow on processing we do (eg. Machine Learning)
   output_tree = new TTree("tree", "tree");
   if (maxTreeSize > 0) output_tree->SetMaxTreeSize(maxTreeSize);

   // Branches for the NN stuff
   for (auto & kv_pair:nn_map_lvbb){
      lvbbNNs[kv_pair.first] = -1; // initialise to -1
      output_tree->Branch(TString(kv_pair.first), &lvbbNNs[kv_pair.first]);
      if ((doCombined) && (kv_pair.first.find("OddTrained") != std::string::npos)){
      // if (doCombined){
         basename_delim_pos = kv_pair.first.find_last_of("_");
         NN_basename = kv_pair.first.substr(0, basename_delim_pos);
         lvbbNNs[NN_basename + "_Combined"] = -1;
         output_tree->Branch(TString(NN_basename + "_Combined"), &lvbbNNs[NN_basename + "_Combined"]);
      }
   }
   for (auto & kv_pair:nn_map_qqbb){
      qqbbNNs[kv_pair.first] = -1; // initialise to -1
      output_tree->Branch(TString(kv_pair.first), &qqbbNNs[kv_pair.first]);
      if ((doCombined) && (kv_pair.first.find("OddTrained") != std::string::npos)){
      // if (doCombined){
         basename_delim_pos = kv_pair.first.find_last_of("_");
         NN_basename = kv_pair.first.substr(0, basename_delim_pos);
         qqbbNNs[NN_basename + "_Combined"] = -1;
         output_tree->Branch(TString(NN_basename + "_Combined"), &qqbbNNs[NN_basename + "_Combined"]);
      }
   }


   // Branches for the low level (ll) variable stuff
   output_tree->Branch("ll_particle_px", &ll_particle_px);
   output_tree->Branch("ll_particle_py", &ll_particle_py);
   output_tree->Branch("ll_particle_pz", &ll_particle_pz);
   output_tree->Branch("ll_particle_e", &ll_particle_e);
   output_tree->Branch("ll_particle_type", &ll_particle_type);
   output_tree->Branch("ll_particle_tagInfo", &ll_particle_tagInfo);
   output_tree->Branch("ll_particle_recoInclusion", &ll_particle_recoInclusion);
   output_tree->Branch("ll_particle_trueInclusion", &ll_particle_trueInclusion);
   output_tree->Branch("ll_truth_decay_mode", &truth_decay_mode);
   output_tree->Branch("ll_truth_decay_mode_old", &truth_decay_mode_old);
   output_tree->Branch("ll_best_mH", &best_mH);
   output_tree->Branch("ll_best_mWqq", &best_mWqq);
   output_tree->Branch("ll_best_mWlv", &best_mWlv);
   output_tree->Branch("ll_best_mWH_lvbb", &best_mWH_lvbb);
   output_tree->Branch("ll_best_mWH_qqbb", &best_mWH_qqbb);
   output_tree->Branch("ll_lepton_count", &lepton_count);
   // output_tree->Branch("RunNumber", &runNumber);

   // Event metadata
   output_tree->Branch("RunNumber", &runNumber);
   output_tree->Branch("eventNumber", &eventNumber);
   output_tree->Branch("eventWeight", &EventWeight);
   output_tree->Branch("DSID", &dsid);
   // Variables for being able to calculate the weights
   if (not lightWeightMode){
      output_tree->Branch("GenFiltHT", &GenFiltHT);
      output_tree->Branch("weight_pileup", &weight_pileup);
      output_tree->Branch("weight_mc", &weight_mc);
      output_tree->Branch("weight_leptonSF", &weight_leptonSF);
      output_tree->Branch("weight_bTagSF_DL1r_Continuous", &weight_bTagSF_DL1r_Continuous);
      output_tree->Branch("weight_jvt", &weight_jvt);
      output_tree->Branch("luminosity_factor", &luminosity_factor);
      output_tree->Branch("weight_normalise", &weight_normalise);
      output_tree->Branch("TopHeavyFlavorFilterFlag", &TopHeavyFlavorFilterFlag);
      // output_tree->Branch("GenFiltHT", &GenFiltHT);
      output_tree->Branch("xsec", &m_xsec);
      output_tree->Branch("kfac", &m_kfac);
      output_tree->Branch("sumOfMCGenWeights", &m_sumOfMCGenWeights);
   }
   // Global event-level info
   output_tree->Branch("Ntags", &m_NTags);
   output_tree->Branch("Njets", &nJets);
   output_tree->Branch("NFatjets", &nLJets);
   output_tree->Branch("HT", &m_HT);
   output_tree->Branch("HT_bjets", &m_HT_bjets);
   // Low-level variables of interest
   output_tree->Branch("MET", &met_met);
   output_tree->Branch("METPhi", &met_phi);
   output_tree->Branch("Lepton_Pt", &Lepton_Pt);
   output_tree->Branch("Lepton_Eta", &Lepton_Eta);
   output_tree->Branch("Lepton_Phi", &Lepton_Phi);
   output_tree->Branch("Lepton_M", &Lepton_M);
   if (not lightWeightMode){
      output_tree->Branch("min_DeltaPhiJETMET", &m_min_DeltaPhiJETMET);
   }
   //output_tree->Branch("HT_bjets_Lepton_Pt", &m_HT_bjets_Lepton_Pt);
   // Truth-level (ie generator level) info
   if (not lightWeightMode){
      output_tree->Branch("Truth_Higgs_Pt", &Truth_Higgs_Pt);
      output_tree->Branch("Truth_Higgs_Eta", &Truth_Higgs_Eta);
      output_tree->Branch("Truth_Higgs_Phi", &Truth_Higgs_Phi);
      output_tree->Branch("Truth_Higgs_M", &Truth_Higgs_M);
      output_tree->Branch("Truth_Wplus_Pt", &Truth_Wplus_Pt);
      output_tree->Branch("Truth_Wplus_Eta", &Truth_Wplus_Eta);
      output_tree->Branch("Truth_Wplus_Phi", &Truth_Wplus_Phi);
      output_tree->Branch("Truth_Wplus_M", &Truth_Wplus_M);
   }
   output_tree->Branch("truth_W_decay_mode", &truth_W_decay_mode);
   output_tree->Branch("truth_lep_charge", &truth_lep_charge);
   output_tree->Branch("truth_agreement", &truth_agreement);
   output_tree->Branch("lep_charge_agreement", &lep_charge_agreement);
   // Reco-level (after our naive-ish reconstruction algorithm) info
   output_tree->Branch("selection_category", &selection_category);
   output_tree->Branch("combined_category", &combined_category);
   if (not lightWeightMode){
      output_tree->Branch("mH", &m_mH);
   }
   output_tree->Branch("mWT", &m_mWT);
   output_tree->Branch("mVH_qqbb", &m_mVH_qqbb); // reconstructed charged Higgs mass if the W is taken to decay hadronically
   output_tree->Branch("mVH_lvbb", &m_mVH_lvbb); // reconstructed charged Higgs mass if the W is taken to decay leptonically
   if (not lightWeightMode){
      output_tree->Branch("mVH_qqbb_WFromSmallRJets", &m_mVH_qqbb_WFromSmallRJets); // reconstructed charged Higgs mass if the W is taken to decay hadronically into a pair of small-R jets
      output_tree->Branch("mVH_qqbb_WFromLargeRJet", &m_mVH_qqbb_WFromLargeRJet); // reconstructed charged Higgs mass if the W is taken to decay hadronically into a large-R jet
   }
   output_tree->Branch("H_mass", &m_H_mass); // Mass of reconstructed 125GeV Higgs
   output_tree->Branch("H_phi", &m_H_phi); // phi  of reconstructed 125GeV Higgs
   output_tree->Branch("H_eta", &m_H_eta); // eta  of reconstructed 125GeV Higgs
   output_tree->Branch("H_Pt", &m_H_Pt); // Pt of reconstructed 125GeV Higgs
   output_tree->Branch("Whad_mass", &m_Whad_mass); // mass of reconstructed hadroincally decaying W boson
   output_tree->Branch("Whad_phi", &m_Whad_phi); // phi of reconstructed hadroincally decaying W boson
   output_tree->Branch("Whad_eta", &m_Whad_eta); // eta of reconstructed hadroincally decaying W boson
   output_tree->Branch("Whad_Pt", &m_Whad_Pt); // Pt of reconstructed hadroincally decaying W boson
   if (not lightWeightMode){
      output_tree->Branch("Whad_FromSmallRJets_mass", &m_Whad_FromSmallRJets_mass); // mass of reconstructed hadroincally decaying W boson IF it is reconstructed using a pair of small-R jets
      output_tree->Branch("Whad_FromSmallRJets_phi", &m_Whad_FromSmallRJets_phi); // phi of reconstructed hadroincally decaying W boson IF it is reconstructed using a pair of small-R jets
      output_tree->Branch("Whad_FromSmallRJets_eta", &m_Whad_FromSmallRJets_eta); // eta of reconstructed hadroincally decaying W boson IF it is reconstructed using a pair of small-R jets
      output_tree->Branch("Whad_FromSmallRJets_Pt", &m_Whad_FromSmallRJets_Pt); // Pt of reconstructed hadroincally decaying W boson IF it is reconstructed using a pair of small-R jets
      output_tree->Branch("Whad_FromLargeRJet_mass", &m_Whad_FromLargeRJet_mass); // mass of reconstructed hadroincally decaying W boson IF it is reconstructed using a single large-R jet
      output_tree->Branch("Whad_FromLargeRJet_phi", &m_Whad_FromLargeRJet_phi); // phi of reconstructed hadroincally decaying W boson IF it is reconstructed using a single large-R jet
      output_tree->Branch("Whad_FromLargeRJet_eta", &m_Whad_FromLargeRJet_eta); // eta of reconstructed hadroincally decaying W boson IF it is reconstructed using a single large-R jet
      output_tree->Branch("Whad_FromLargeRJet_Pt", &m_Whad_FromLargeRJet_Pt); // Pt of reconstructed hadroincally decaying W boson IF it is reconstructed using a single large-R jet
   }
   output_tree->Branch("Wlep_mass", &m_Wlep_mass); // mass of reconstructed leptonically decaying W boson
   output_tree->Branch("Wlep_phi", &m_Wlep_phi); // phi of reconstructed leptonically decaying W boson
   output_tree->Branch("Wlep_eta", &m_Wlep_eta); // eta of reconstructed leptonically decaying W boson
   output_tree->Branch("Wlep_Pt", &m_Wlep_Pt); // Pt of reconstructed leptonically decaying W boson
   //output_tree->Branch("MaxMVA_Response", &m_MaxMVA_Response);
   //output_tree->Branch("pTH_over_mVH", &m_pTH_over_mVH);
   output_tree->Branch("bTagCategory", &m_bTagCategory);
   if (not lightWeightMode){
      output_tree->Branch("mass_resolution_qqbb", &m_mass_resolution_qqbb);
      output_tree->Branch("mass_resolution_lvbb", &m_mass_resolution_lvbb);
   }
   //output_tree->Branch("MET_over_sqrtHT", &m_MET_over_sqrtHT);
   output_tree->Branch("NTags_trkJ", &m_NTags_trkJ);
   if (not lightWeightMode){
      output_tree->Branch("ljet_Xbb2020v3_Higgs", &xbb_tag_higgsJet_value);
   }
   output_tree->Branch("Xbb_variable_FJet_Higgs", &Xbb_variable_FJet_Higgs);
   output_tree->Branch("Xbb_variable_FJet_WHad", &Xbb_variable_FJet_WHad);
   output_tree->Branch("LepEnergyFracHad", &m_LepEnergyFrac_qqbb);
   output_tree->Branch("LepEnergyFracLep", &m_LepEnergyFrac_lvbb);
   output_tree->Branch("DeltaPhi_H_Lep", &m_DeltaPhi_H_Lep);
   output_tree->Branch("DeltaPhi_H_MET", &m_DeltaPhi_H_MET);
   output_tree->Branch("DeltaPhi_W_hadronic_Lep", &m_DeltaPhi_W_hadronic_Lep);
   output_tree->Branch("DeltaPhi_W_hadronic_MET", &m_DeltaPhi_W_hadronic_MET);
   if (not lightWeightMode){
      output_tree->Branch("DeltaPhi_HW_leptonic", &m_DeltaPhi_HW_hadronic);
   }
      output_tree->Branch("DeltaR_HW_hadronic", &m_DeltaR_HW_hadronic);
      output_tree->Branch("DeltaR_HW_leptonic", &m_DeltaR_HW_leptonic);
   
   output_tree->Branch("deltaR_LH", &m_deltaR_LH);
   output_tree->Branch("deltaR_LWhad", &m_deltaR_LWhad);
   output_tree->Branch("deltaEta_HWhad", &m_deltaEta_HWhad);
   output_tree->Branch("deltaPhi_HWhad", &m_deltaPhi_HWhad);
   output_tree->Branch("deltaEta_HWlep", &m_deltaEta_HWlep);
   output_tree->Branch("deltaPhi_HWlep", &m_deltaPhi_HWlep);
   output_tree->Branch("ratio_Wpt_mVH_qqbb", &ratio_Wpt_mVH_qqbb);
   output_tree->Branch("ratio_Wpt_mVH_lvbb", &ratio_Wpt_mVH_lvbb);
   output_tree->Branch("ratio_Hpt_mVH_qqbb", &ratio_Hpt_mVH_qqbb);
   output_tree->Branch("ratio_Hpt_mVH_lvbb", &ratio_Hpt_mVH_lvbb);
   if (not lightWeightMode){
      output_tree->Branch("pass_Merged_CR", &pass_sel["Merged_CR"]);
      output_tree->Branch("pass_Merged_SR", &pass_sel["Merged_SR"]);
   }
   output_tree->Branch("mTop_lepto", &m_mTop_lepto);
   output_tree->Branch("mTop_hadro", &m_mTop_hadro);
   output_tree->Branch("mTop_best", &m_mTop_best);
   output_tree->Branch("nTagsTrkJets", &m_NTags_trkJ);
   output_tree->Branch("nTagsCaloJets", &m_NTags_caloJ);
   if (not lightWeightMode){
      output_tree->Branch("diff_trk_calo_btags", &m_diff_trk_calo_btags);
      output_tree->Branch("diff_trk_calo_jets", &m_diff_trk_calo_jets);
   }
   output_tree->Branch("nTrkTagsOutside", &m_nTrkTagsOutside); // Number of track b-tags outside the 125Higgs and the W boson identified as decaying from charged Higgs
   output_tree->Branch("nTrkTagsInW", &m_nTrkTagsInW); // Number of track b-tags inside the W boson identified as decaying from charged Higgs
   output_tree->Branch("nTrkTagsInH", &m_nTrkTagsInH); // Number of track b-tags inside the 125Higgs identified as decaying from charged Higgs
   if (not lightWeightMode){
      output_tree->Branch("nTrkTagsOutside_WFromsmallR", &m_nTrkTagsOutside_smallR); // Number of track b-tags outside the 125Higgs and the W boson identified as decaying from charged Higgs if the W boson is reconstructed from a pair of small-R jets
      output_tree->Branch("nTrkTagsInW_WFromsmallR", &m_nTrkTagsInW_smallR); // Number of track b-tags inside the W boson identified as decaying from charged Higgs if the W boson is reconstructed from a pair of small-R jets
      output_tree->Branch("nTrkTagsInH_WFromsmallR", &m_nTrkTagsInH_smallR); // Number of track b-tags inside the 125Higgs identified as decaying from charged Higgs if the W boson is reconstructed from a pair of small-R jets
      output_tree->Branch("nTrkTagsOutside_WFromlargeR", &m_nTrkTagsOutside_largeR); // Number of track b-tags outside the 125Higgs and the W boson identified as decaying from charged Higgs if the W boson is reconstructed from a large-R jet
      output_tree->Branch("nTrkTagsInW_WFromlargeR", &m_nTrkTagsInW_largeR); // Number of track b-tags inside the W boson identified as decaying from charged Higgs if the W boson is reconstructed from a large-R jet
      output_tree->Branch("nTrkTagsInH_WFromlargeR", &m_nTrkTagsInH_largeR); // Number of track b-tags inside the 125Higgs identified as decaying from charged Higgs if the W boson is reconstructed from a large-R jet
   }
   if (lowlevel_output_mode){
      output_tree->Branch("smallJets_Pt", &Jets_Pt);
      output_tree->Branch("smallJets_Eta", &Jets_Eta);
      output_tree->Branch("smallJets_Phi", &Jets_Phi);
      output_tree->Branch("smallJets_M", &Jets_M);
      output_tree->Branch("smallJets_tagWeightBinDL1rContinuous", &Jets_tagWeightBinDL1rContinuous);
      output_tree->Branch("largeJets_Pt", &FJets_Pt);
      output_tree->Branch("largeJets_Eta", &FJets_Eta);
      output_tree->Branch("largeJets_Phi", &FJets_Phi);
      output_tree->Branch("largeJets_M", &FJets_M);
      output_tree->Branch("largeJets_DXbb", &FJets_DXbb);
   }
   


   Init(tree, SampleName, ExpUncertaintyName);
}

// TODO Work out what the below definition does? When is it used?
EventLoop::EventLoop(TTree *tree, 
                     TString sampleName, 
                     TString ExpUncertaintyName, 
                     TString WP, 
                     bool UnweightedCutFlow,
                     bool WeightedCutFlow, 
                     bool UnweightedAltCutFlow, 
                     bool WeightedAltCutFlow, 
                     Float_t hmlb, 
                     Float_t hmub, 
                     Float_t wmlb,
                     Float_t wmub, 
                     Float_t met_pt_min, 
                     Float_t lep_pt_min, 
                     Float_t higgs_pt_min, 
                     Float_t W_leptonic_pt_min, 
                     Float_t lep_SMHiggs_angle_min,
                     Float_t lep_SMHiggs_angle_max,
                     Float_t lep_W_hadronic_angle_min, 
                     Float_t hw_angle) : fChain(tree),
                                          hmlb(hmlb),
                                          hmub(hmub),
                                          wmlb(wmlb),
                                          wmub(wmub),
                                          met_pt_min(met_pt_min),
                                          lep_pt_min(lep_pt_min),
                                          higgs_pt_min(higgs_pt_min),
                                          W_leptonic_pt_min(W_leptonic_pt_min),
                                          lep_SMHiggs_angle_min(lep_SMHiggs_angle_min),
                                          lep_SMHiggs_angle_max(lep_SMHiggs_angle_max),
                                          lep_W_hadronic_angle_min(lep_W_hadronic_angle_min),
                                          hw_angle(hw_angle)
{

   // if parameter tree is not specified (or zero), connect the file
   // used to generate this class and read the Tree.
   if (tree == 0)
   {
      std::cerr << "Error in EventLoop::EventLoop(): tree is nullptr" << std::endl;
      return;
   }
   //m_btagCut_value_trkJets = -1.;
   //m_btagCategoryBin_trkJets = -1;
   m_btagCategoryBin = 10; // If not set then  don't have a well defined working point. Value of 10 means that we fail ALL Btags
   if (WP == "85p")
   {
      //m_btagCut_value_trkJets = 0.05;
      //m_btagCategoryBin_trkJets = 2; // to replace above line. TODO is this right??
      //m_btagCut_value_CaloJets = 0.11;
      m_btagCategoryBin = 2;
   }
   if (WP == "77p")
   {
      //m_btagCut_value_trkJets = 0.58;
      //m_btagCategoryBin_trkJets = 3; // to replace above line. TODO is this right??
      //m_btagCut_value_CaloJets = 0.64;
      m_btagCategoryBin = 3;
   }
   if (WP == "70p")
   {
      //m_btagCut_value_trkJets = 0.79;
      //m_btagCategoryBin_trkJets = 4; // to replace above line. TODO is this right??
      //m_btagCut_value_CaloJets = 0.83;
      m_btagCategoryBin = 4;
   }
   if (WP == "60p")
   {
      //m_btagCut_value_trkJets = 0.92;
      //m_btagCategoryBin_trkJets = 5; // to replace above line. TODO is this right??
      //m_btagCut_value_CaloJets = 0.94;
      m_btagCategoryBin = 5;
   }

   std::cout << "Below line is meaningless; I upaded btag cut fro track jets to use bin but didn't update the below line." << std::endl;
   //std::cout << "Using WP = " << WP << " corresponding to w_{MVA} bin > " << m_btagCategoryBin_trkJets << std::endl;
   std::cout << "Using WP = " << WP << " corresponding to w_{MVA} bin > " << m_btagCategoryBin << std::endl;

   Init(tree, sampleName, ExpUncertaintyName);
}

// TODO I had to remove the 'const' before the argument "const CutFlowType &cutVariable" from this, to get it to build. Would be nice to put this back, as we probably want to make sure we don't change cutVariable in here.
void EventLoop::CutFlowParser(std::ofstream &File, CutFlowType &cutVariable, const std::string cutName, bool XUnweightedCutFlow, bool XWeightedCutFlow)
{
   if (XUnweightedCutFlow == true)
   {
      File << std::fixed;
      File << "Unweighted" << cutName << "," 
            << cutVariable.unweighted_vals["both_channels"]["zeroTags"] << "," 
            << cutVariable.unweighted_vals["both_channels"]["oneTags"] << "," 
            << cutVariable.unweighted_vals["both_channels"]["twoTags"] << "," 
            << cutVariable.unweighted_vals["both_channels"]["threeTags"] << "," 
            << cutVariable.unweighted_vals["both_channels"]["fourPlusTags"] << ","
            << cutVariable.both_UnweightedTotal()
            << ","
            << cutVariable.unweighted_vals["jjbb_channel"]["zeroTags"] << "," 
            << cutVariable.unweighted_vals["jjbb_channel"]["oneTags"] << ","
            << cutVariable.unweighted_vals["jjbb_channel"]["twoTags"] << "," 
            << cutVariable.unweighted_vals["jjbb_channel"]["threeTags"] << ","
            << cutVariable.unweighted_vals["jjbb_channel"]["fourPlusTags"] << "," 
            << cutVariable.jjbb_UnweightedTotal() 
            << "," 
            << cutVariable.unweighted_vals["lvbb_channel"]["zeroTags"] << "," 
            << cutVariable.unweighted_vals["lvbb_channel"]["oneTags"] << "," 
            << cutVariable.unweighted_vals["lvbb_channel"]["twoTags"] << "," 
            << cutVariable.unweighted_vals["lvbb_channel"]["threeTags"] << "," 
            << cutVariable.unweighted_vals["lvbb_channel"]["fourPlusTags"] << ","
            << cutVariable.lvbb_UnweightedTotal()
            << "\n";
   }
   if (XWeightedCutFlow == true)
   {
      File << std::fixed;
      File << cutName << ","
            << cutVariable.weighted_vals["both_channels"]["zeroTags"] << "," 
            << cutVariable.weighted_vals["both_channels"]["oneTags"] << "," 
            << cutVariable.weighted_vals["both_channels"]["twoTags"] << "," 
            << cutVariable.weighted_vals["both_channels"]["threeTags"] << "," 
            << cutVariable.weighted_vals["both_channels"]["fourPlusTags"] << ","
            << cutVariable.both_WeightedTotal()
            << "," 
            << cutVariable.weighted_vals["jjbb_channel"]["zeroTags"] << "," 
            << cutVariable.weighted_vals["jjbb_channel"]["oneTags"] << "," 
            << cutVariable.weighted_vals["jjbb_channel"]["twoTags"] << "," 
            << cutVariable.weighted_vals["jjbb_channel"]["threeTags"] << ","
            << cutVariable.weighted_vals["jjbb_channel"]["fourPlusTags"] << "," 
            << cutVariable.jjbb_WeightedTotal() 
            << "," 
            << cutVariable.weighted_vals["lvbb_channel"]["zeroTags"] << ","
            << cutVariable.weighted_vals["lvbb_channel"]["oneTags"] << ","
            << cutVariable.weighted_vals["lvbb_channel"]["twoTags"] << "," 
            << cutVariable.weighted_vals["lvbb_channel"]["threeTags"] << "," 
            << cutVariable.weighted_vals["lvbb_channel"]["fourPlusTags"] << ","
            << cutVariable.lvbb_WeightedTotal()
            << "\n";
   }
}

EventLoop::~EventLoop()
{
   if (debugMode) std::cout << "\t" << "Entering ~EventLoop()" << std::endl;
   if (UnweightedCutFlow == true || WeightedCutFlow == true)
   {
      m_cutFlowFileStream.open(m_cutFlowFileName);
      // Cutflow variables will be written to a csv file with headers 'subset.bTagStrategy', where subset indicates a particular subset of events (eg. 'all', 'truth_jjbb', etc. these do not have to be disjoint) and the bTagStrategy indicates the number of btags.
      m_cutFlowFileStream << "CutVariableName,all.zeroTags,all.oneTags,all.twoTags,all.threeTags,all.fourPlusTags,all.allTags,truth_jjbb.zeroTags,truth_jjbb.oneTags,truth_jjbb.twoTags,truth_jjbb.threeTags,truth_jjbb.fourPlusTags,truth_jjbb.allTags,truth_lvbb.zeroTags,truth_lvbb.oneTags,truth_lvbb.twoTags,truth_lvbb.threeTags,truth_lvbb.fourPlusTags,truth_lvbb.allTags\n";
      m_cutFlowParametersFileStream.open(m_cutFlowParametersFileName);
      std::stringstream ss; // This variable will help us format the strings

      // Write the cut values and the cut parameter names out to file (in the correct order)
      CutFlowParser(m_cutFlowFileStream, m_TotalEvents, "Total Events", UnweightedCutFlow, WeightedCutFlow);
      m_cutFlowParametersFileStream << "Total Events,";

      if (min_n_fat_jets_cut_on)
      {
         CutFlowParser(m_cutFlowFileStream, m_MinNFatJetsCutFlow, "At least " + std::to_string(min_n_fat_jets) + " Fat Jets", UnweightedCutFlow, WeightedCutFlow);
         m_cutFlowParametersFileStream << "At least " + std::to_string(min_n_fat_jets) + " Fat Jets,";
      }
      if (max_n_fat_jets_cut_on)
      {
         CutFlowParser(m_cutFlowFileStream, m_MaxNFatJetsCutFlow, "At most " + std::to_string(max_n_fat_jets) + " Fat Jets", UnweightedCutFlow, WeightedCutFlow);
         m_cutFlowParametersFileStream << "At most " + std::to_string(max_n_fat_jets) + " Fat Jets,";
      }
      if (met_pt_min_cut_on) 
      {
         ss.str(std::string());
         ss << std::fixed << std::setprecision(1) << met_pt_min;
         std::string mystring = ss.str();
         CutFlowParser(m_cutFlowFileStream, m_METCutFlow, "MET > " + mystring, UnweightedCutFlow, WeightedCutFlow);
         m_cutFlowParametersFileStream << "MET > " + mystring + ",";
      }
      if(lep_pt_min_cut_on) 
      {
         ss.str(std::string());
         ss << std::fixed << std::setprecision(0) << lep_pt_min;
         std::string mystring = ss.str();
         CutFlowParser(m_cutFlowFileStream, m_LeptonPtCutFlow, "Lep Pt > " + mystring, UnweightedCutFlow, WeightedCutFlow);
         m_cutFlowParametersFileStream << "Lep Pt > " + mystring + ",";
      }
      if(W_leptonic_pt_min_cut_on) 
      {
         ss.str(std::string());
         ss << std::fixed << std::setprecision(0) << W_leptonic_pt_min/1000;
         std::string mystring = ss.str();
         CutFlowParser(m_cutFlowFileStream, m_WPtCutFlow, "W^{lep} Pt > " + mystring + "GeV", UnweightedCutFlow, WeightedCutFlow);
         m_cutFlowParametersFileStream << "W^{lep} Pt > " + mystring + "GeV" + ",";
      }
      if(category_cut_on) 
      {
         std::string mystring = "";
         for(int cat : chosen_categories){
            mystring.append(std::to_string(cat));
            mystring.append(" ");
         }
         CutFlowParser(m_cutFlowFileStream, m_selectionCategoryCutFlow, "selectionCategory in " + mystring, UnweightedCutFlow, WeightedCutFlow);
         m_cutFlowParametersFileStream << "selectionCategory in " + mystring + ",";
      }
      //CutFlowParser(m_cutFlowFileStream, m_ChannelFlexCutFlow, "jjbb_OR_lvbb_rejected", UnweightedCutFlow, WeightedCutFlow);
      if (higgs_pt_min_cut_on)
      {
         ss.str(std::string());
         ss << std::fixed << std::setprecision(2) << higgs_pt_min;
         std::string mystring = ss.str();
         CutFlowParser(m_cutFlowFileStream, m_HiggsPtCutFlow, "SM Higgs Pt > " + mystring, UnweightedCutFlow, WeightedCutFlow);
         m_cutFlowParametersFileStream << "SM Higgs Pt > " + mystring + ",";
      }
      if (lep_SMHiggs_angle_min_cut_on) 
      {
         ss.str(std::string());
         ss << std::fixed << std::setprecision(2) << lep_SMHiggs_angle_min;
         std::string mystring = ss.str();
         CutFlowParser(m_cutFlowFileStream, m_Higgs_LeptonAngleCutflow, "Higgs Lep Angle > " + mystring, UnweightedCutFlow, WeightedCutFlow);
         m_cutFlowParametersFileStream << "Higgs Lep Angle > " + mystring + ",";
      }
      if (lep_SMHiggs_angle_max_cut_on) 
      {
         ss.str(std::string());
         ss << std::fixed << std::setprecision(2) << lep_SMHiggs_angle_max;
         std::string mystring = ss.str();
         CutFlowParser(m_cutFlowFileStream, m_Higgs_MaxLeptonAngleCutflow, "Higgs Lep Angle < " + mystring, UnweightedCutFlow, WeightedCutFlow);
         m_cutFlowParametersFileStream << "Higgs Lep Angle < " + mystring + ",";
      }
      if (lep_W_hadronic_angle_min_cut_on)
      {
         ss.str(std::string());
         ss << std::fixed << std::setprecision(2) << lep_W_hadronic_angle_min;
         std::string mystring = ss.str();
         CutFlowParser(m_cutFlowFileStream, m_W_hadronic_LeptonAngleCutflow, "Whad Lep Angle > " + mystring, UnweightedCutFlow, WeightedCutFlow);
         m_cutFlowParametersFileStream << "Whad Lep Angle > " + mystring + ",";
      }
      if (hw_angle_cut_on) 
      {
         ss.str(std::string());
         ss << std::fixed << std::setprecision(2) << hw_angle;
         std::string mystring = ss.str();
         CutFlowParser(m_cutFlowFileStream, m_Higgs_WplusAngleCutflow, "Higgs W+ Angle > " + mystring, UnweightedCutFlow, WeightedCutFlow);
         m_cutFlowParametersFileStream << "Higgs W+ Angle > " + mystring + ",";
      }
      if (status_W_cut_on)
      {
         CutFlowParser(m_cutFlowFileStream, m_PositiveLepWCutflow, "W+ Boson found", UnweightedCutFlow, WeightedCutFlow);
         m_cutFlowParametersFileStream << "W+ Boson found,";
      }
      //if (solo_jet_ptv_cut_on  && min_n_fat_jets == 1) TODO maybe add conditions based upon how many fat_jets there are. Or just rejig how the different n_fat_jets cases are handled. Maybe this no longer needs doing?
      if (wm_cut_on)
      {
         ss.str(std::string());
         ss << std::fixed << std::setprecision(0) << wmlb;
         std::string mystring1 = ss.str();
         ss.str(std::string());
         ss << std::fixed << std::setprecision(0) << wmub;
         std::string mystring2 = ss.str();
         CutFlowParser(m_cutFlowFileStream, m_WplusMassCutFlow, mystring1 + " < W+ Mass < " + mystring2, UnweightedCutFlow, WeightedCutFlow);
         m_cutFlowParametersFileStream << mystring1 + " < W+ Mass < " + mystring2 + ",";
      }
      //if (true)
      //{
      ss.str(std::string());
      ss << std::fixed << std::setprecision(0) << hmlb;
      std::string mystring1 = ss.str();
      ss.str(std::string());
      ss << std::fixed << std::setprecision(0) << hmub;
      std::string mystring2 = ss.str();
      // CRSR is a string which we add to the start of the variable name to identify that this is used to split Control Region/Signal Region, rather than to remove events
      CutFlowParser(m_cutFlowFileStream, m_HiggsMassCutFlow, "CRSR" + mystring1 + " < SM Higgs Mass < " + mystring2, UnweightedCutFlow, WeightedCutFlow);
      m_cutFlowParametersFileStream << "CRSR" + mystring1 + " < SM Higgs Mass < " + mystring2 + ",";
      //}




      if (UnweightedCutFlow == true)
      {
         m_cutFlowFileStream << "INFOnoFatJets" << "=" 
                              << m_noJets.jjbb_UnweightedTotal() << "," 
                              << m_noJets.lvbb_UnweightedTotal() << "," 
                              << m_noJets.both_UnweightedTotal() << "\n";
      }
      if (WeightedCutFlow == true)
      {
         m_cutFlowFileStream << "INFOWeightednoFatJets" << "=" 
                              << m_noJets.jjbb_WeightedTotal() << "," 
                              << m_noJets.lvbb_WeightedTotal() << "," 
                              << m_noJets.both_WeightedTotal() << "\n";
      }
      m_cutFlowFileStream.close();
      m_cutFlowParametersFileStream.close();
   }

   if (UnweightedAltCutFlow == true || WeightedAltCutFlow == true) //This is a place for the alternate cutflow file
   {
      m_cutFlowFileStreamAlt.open(m_cutFlowFileNameAlt);
      if (UnweightedAltCutFlow == true)
      {
         m_cutFlowFileStreamAlt << "noFatJets"
                                << "=" << m_noJets.jjbb_UnweightedTotal() << "," << m_noJets.lvbb_UnweightedTotal() << "\n";
      }
      if (WeightedAltCutFlow == true)
      {
         m_cutFlowFileStreamAlt << "Weighted"
                                << "noFatJets"
                                << "=" << m_noJets.jjbb_WeightedTotal() << "," << m_noJets.lvbb_WeightedTotal() << "\n";
      }
      m_cutFlowFileStreamAlt.close();
   }
   
   // Delete the extra variables we made for the output tree
   // Update: 'delete' actually seems to want pointers to these
   // TODO work out what needs to be done here to prevent memory leak
   /*delete met_met;
   delete met_phi;
   delete Lepton_Eta;
   delete Lepton_Pt;
   delete Lepton_Phi;
   delete m_HT_bjets_Lepton_Pt;
   delete m_pTH;
   delete m_pTH_over_mVH;
   delete m_mH;
   delete m_mass_resolution_qqbb;
   delete m_mass_resolution_lvbb;
   delete m_MET_over_sqrtHT;
   delete m_pTW_leptonic;
   delete m_mW_leptonic;
   delete m_pTW_hadronic;
   delete m_mW_hadronic;
   delete m_LepEnergyFrac_qqbb;
   delete m_LepEnergyFrac_lvbb;
   delete m_deltaR_LH;
   delete m_deltaR_LWhad;
   delete m_deltaEta_HWhad;
   delete m_deltaPhi_HWhad;
   delete m_deltaEta_HWlep;
   delete m_deltaPhi_HWlep;
   delete ratio_Wpt_mVH_qqbb;
   delete ratio_Wpt_mVH_lvbb;
   delete ratio_Hpt_mVH_qqbb;
   delete ratio_Hpt_mVH_lvbb;
   delete truth_W_decay_mode;
   //delete nJets;
   //delete nLjets;
   */
   // Delete the output tree itself
   output_tree->Write();
   delete output_tree;

   if (!fChain) return;
   delete fChain->GetCurrentFile();
   if (debugMode) std::cout << "\t" << "Leaving ~EventLoop()" << std::endl;
}

Int_t EventLoop::GetEntry(Long64_t entry)
{
// Read contents of entry.
   if (!fChain) return 0;
   return fChain->GetEntry(entry);
}
Long64_t EventLoop::LoadTree(Long64_t entry)
{
// Set the environment to read one entry
   if (!fChain) return -5;
   Long64_t centry = fChain->LoadTree(entry);
   if (centry < 0) return centry;
   if (fChain->GetTreeNumber() != fCurrent) {
      fCurrent = fChain->GetTreeNumber();
      Notify();
   }
   return centry;
}

void EventLoop::Init(TTree *tree, TString sampleName, TString ExpUncertaintyName)
{
   // The Init() function is called when the selector needs to initialize
   // a new tree or chain. Typically here the branch addresses and branch
   // pointers of the tree will be set.
   // It is normally not necessary to make changes to the generated
   // code, but the routine can be extended by the user if needed.
   // Init() will be called many times when running on PROOF
   // (once per file to be processed).
   if (debugMode) std::cout << "" << "Entering EventLoop::Init function" << std::endl;
   processName = sampleName;
   m_NeutrinoBuilder = new NeutrinoBuilder("MeV");
   if (debugMode) std::cout << "\t" << "is_signal_sample = " << is_signal_sample << std::endl;
   
   // Now get the dsid.
   if (weights_included) { // For now, only require this when we have MC samples, not data
      tree->SetBranchAddress("mcChannelNumber", &dsid);
      tree->GetEntry(0);
   }
   else {
      dsid = 0;
   }
   if (debugMode) std::cout << "\t" << "In EventLoop::Init function. Updated dsid to: " << dsid << std::endl;

   // Check if we are signal or sample
   is_signal_sample = false;
   is_signal_sample = std::count(SIGNAL_DSIDS.begin(), SIGNAL_DSIDS.end(), dsid) > 0;

   m_UncNames = {""};
   // The selections in mySel are no longer disjoint.
   //    - the Merged_SR_All and Merged_CR_All will be disjoint and cover all possibilities (ie span the set). They will be filled for all sample types
   //    - the ones containing 'truth' will only be filled by the MC signal samples (ie those for which truth info is available)
   mySel = {"Merged_SR",
            "Merged_CR",
            "Merged_SR_subset_truth_jjbb",
            "Merged_CR_subset_truth_jjbb",
            "Merged_SR_subset_truth_lvbb",
            "Merged_CR_subset_truth_lvbb"}; // Merged is synonymous with 'Boosted channel'; it means the SM Higgs and W+ are merged into the same jet

   // Set object pointer
   mc_generator_weights = 0;
   weight_bTagSF_DL1r_Continuous_eigenvars_B_up = 0;
   weight_bTagSF_DL1r_Continuous_eigenvars_C_up = 0;
   weight_bTagSF_DL1r_Continuous_eigenvars_Light_up = 0;
   weight_bTagSF_DL1r_Continuous_eigenvars_B_down = 0;
   weight_bTagSF_DL1r_Continuous_eigenvars_C_down = 0;
   weight_bTagSF_DL1r_Continuous_eigenvars_Light_down = 0;
   weight_trackjet_bTagSF_DL1r_Continuous_eigenvars_B_up = 0;
   weight_trackjet_bTagSF_DL1r_Continuous_eigenvars_C_up = 0;
   weight_trackjet_bTagSF_DL1r_Continuous_eigenvars_Light_up = 0;
   weight_trackjet_bTagSF_DL1r_Continuous_eigenvars_B_down = 0;
   weight_trackjet_bTagSF_DL1r_Continuous_eigenvars_C_down = 0;
   weight_trackjet_bTagSF_DL1r_Continuous_eigenvars_Light_down = 0;
   el_pt = 0;
   el_eta = 0;
   el_cl_eta = 0;
   el_phi = 0;
   el_e = 0;
   el_charge = 0;
   el_topoetcone20 = 0;
   el_ptvarcone20 = 0;
   el_isTight = 0;
   el_CF = 0;
   el_d0sig = 0;
   el_delta_z0_sintheta = 0;
   mu_pt = 0;
   mu_eta = 0;
   mu_phi = 0;
   mu_e = 0;
   mu_charge = 0;
   mu_topoetcone20 = 0;
   mu_ptvarcone30 = 0;
   mu_isTight = 0;
   mu_d0sig = 0;
   mu_delta_z0_sintheta = 0;
   jet_pt = 0;
   jet_eta = 0;
   jet_phi = 0;
   jet_e = 0;
   jet_jvt = 0;
   jet_truthflav = 0;
   jet_truthflavExtended = 0;
   jet_isbtagged_DL1r_60 = 0;
   jet_isbtagged_DL1r_70 = 0;
   jet_isbtagged_DL1r_77 = 0;
   jet_isbtagged_DL1r_85 = 0;
   jet_tagWeightBin_DL1r_Continuous = 0;
   jet_DL1r = 0;
   ljet_pt = 0;
   ljet_eta = 0;
   ljet_phi = 0;
   ljet_e = 0;
   ljet_m = 0;
   ljet_truthLabel = 0;
   tjet_pt = 0;
   tjet_eta = 0;
   tjet_phi = 0;
   tjet_e = 0;
   tjet_tagWeightBin_DL1r_Continuous = 0;
   tjet_DL1r = 0;
   ljet_C2 = 0;
   ljet_D2 = 0;
   ljet_Xbb2020v3_Higgs = 0;
   ljet_Xbb2020v3_QCD = 0;
   ljet_Xbb2020v3_Top = 0;
   ljet_muonCorrectedEta = 0;
   ljet_muonCorrectedMass = 0;
   ljet_muonCorrectedPhi = 0;
   ljet_muonCorrectedPt = 0;
   truth_pt = 0;
   truth_eta = 0;
   truth_phi = 0;
   truth_m = 0;
   truth_pdgid = 0;
   truth_status = 0;
   truth_barcode = 0;
   truth_tthbb_info = 0;
   truth_jet_pt = 0;
   truth_jet_eta = 0;
   truth_jet_phi = 0;
   truth_jet_m = 0;
   // Set branch addresses and branch pointers
   if (!tree) return; // this line has been commented out in Zahaab's code; TODO work out why?
   fChain = tree; // this line has been commented out in Zahaab's code; TODO work out why?
   fCurrent = -1;
   fChain->SetMakeClass(1);
   if (weights_included) // These will only exist if it's an MC sample, rather than a data sample
   {
      fChain->SetBranchAddress("mc_generator_weights", &mc_generator_weights, &b_mc_generator_weights);
      fChain->SetBranchAddress("weight_mc", &weight_mc, &b_weight_mc);
      fChain->SetBranchAddress("GenFiltHT", &GenFiltHT, &b_GenFiltHT);
      fChain->SetBranchAddress("weight_pileup", &weight_pileup, &b_weight_pileup);
      fChain->SetBranchAddress("weight_leptonSF", &weight_leptonSF, &b_weight_leptonSF);
      fChain->SetBranchAddress("weight_globalLeptonTriggerSF", &weight_globalLeptonTriggerSF, &b_weight_globalLeptonTriggerSF);
      fChain->SetBranchAddress("weight_oldTriggerSF", &weight_oldTriggerSF, &b_weight_oldTriggerSF);
      fChain->SetBranchAddress("weight_bTagSF_DL1r_Continuous", &weight_bTagSF_DL1r_Continuous, &b_weight_bTagSF_DL1r_Continuous);
      fChain->SetBranchAddress("weight_trackjet_bTagSF_DL1r_Continuous", &weight_trackjet_bTagSF_DL1r_Continuous, &b_weight_trackjet_bTagSF_DL1r_Continuous);
      fChain->SetBranchAddress("weight_jvt", &weight_jvt, &b_weight_jvt);
      fChain->SetBranchAddress("weight_pileup_UP", &weight_pileup_UP, &b_weight_pileup_UP);
      fChain->SetBranchAddress("weight_pileup_DOWN", &weight_pileup_DOWN, &b_weight_pileup_DOWN);
      fChain->SetBranchAddress("weight_leptonSF_EL_SF_Trigger_UP", &weight_leptonSF_EL_SF_Trigger_UP, &b_weight_leptonSF_EL_SF_Trigger_UP);
      fChain->SetBranchAddress("weight_leptonSF_EL_SF_Trigger_DOWN", &weight_leptonSF_EL_SF_Trigger_DOWN, &b_weight_leptonSF_EL_SF_Trigger_DOWN);
      fChain->SetBranchAddress("weight_leptonSF_EL_SF_Reco_UP", &weight_leptonSF_EL_SF_Reco_UP, &b_weight_leptonSF_EL_SF_Reco_UP);
      fChain->SetBranchAddress("weight_leptonSF_EL_SF_Reco_DOWN", &weight_leptonSF_EL_SF_Reco_DOWN, &b_weight_leptonSF_EL_SF_Reco_DOWN);
      fChain->SetBranchAddress("weight_leptonSF_EL_SF_ID_UP", &weight_leptonSF_EL_SF_ID_UP, &b_weight_leptonSF_EL_SF_ID_UP);
      fChain->SetBranchAddress("weight_leptonSF_EL_SF_ID_DOWN", &weight_leptonSF_EL_SF_ID_DOWN, &b_weight_leptonSF_EL_SF_ID_DOWN);
      fChain->SetBranchAddress("weight_leptonSF_EL_SF_Isol_UP", &weight_leptonSF_EL_SF_Isol_UP, &b_weight_leptonSF_EL_SF_Isol_UP);
      fChain->SetBranchAddress("weight_leptonSF_EL_SF_Isol_DOWN", &weight_leptonSF_EL_SF_Isol_DOWN, &b_weight_leptonSF_EL_SF_Isol_DOWN);
      fChain->SetBranchAddress("weight_leptonSF_MU_SF_Trigger_STAT_UP", &weight_leptonSF_MU_SF_Trigger_STAT_UP, &b_weight_leptonSF_MU_SF_Trigger_STAT_UP);
      fChain->SetBranchAddress("weight_leptonSF_MU_SF_Trigger_STAT_DOWN", &weight_leptonSF_MU_SF_Trigger_STAT_DOWN, &b_weight_leptonSF_MU_SF_Trigger_STAT_DOWN);
      fChain->SetBranchAddress("weight_leptonSF_MU_SF_Trigger_SYST_UP", &weight_leptonSF_MU_SF_Trigger_SYST_UP, &b_weight_leptonSF_MU_SF_Trigger_SYST_UP);
      fChain->SetBranchAddress("weight_leptonSF_MU_SF_Trigger_SYST_DOWN", &weight_leptonSF_MU_SF_Trigger_SYST_DOWN, &b_weight_leptonSF_MU_SF_Trigger_SYST_DOWN);
      fChain->SetBranchAddress("weight_leptonSF_MU_SF_ID_STAT_UP", &weight_leptonSF_MU_SF_ID_STAT_UP, &b_weight_leptonSF_MU_SF_ID_STAT_UP);
      fChain->SetBranchAddress("weight_leptonSF_MU_SF_ID_STAT_DOWN", &weight_leptonSF_MU_SF_ID_STAT_DOWN, &b_weight_leptonSF_MU_SF_ID_STAT_DOWN);
      fChain->SetBranchAddress("weight_leptonSF_MU_SF_ID_SYST_UP", &weight_leptonSF_MU_SF_ID_SYST_UP, &b_weight_leptonSF_MU_SF_ID_SYST_UP);
      fChain->SetBranchAddress("weight_leptonSF_MU_SF_ID_SYST_DOWN", &weight_leptonSF_MU_SF_ID_SYST_DOWN, &b_weight_leptonSF_MU_SF_ID_SYST_DOWN);
      fChain->SetBranchAddress("weight_leptonSF_MU_SF_ID_STAT_LOWPT_UP", &weight_leptonSF_MU_SF_ID_STAT_LOWPT_UP, &b_weight_leptonSF_MU_SF_ID_STAT_LOWPT_UP);
      fChain->SetBranchAddress("weight_leptonSF_MU_SF_ID_STAT_LOWPT_DOWN", &weight_leptonSF_MU_SF_ID_STAT_LOWPT_DOWN, &b_weight_leptonSF_MU_SF_ID_STAT_LOWPT_DOWN);
      fChain->SetBranchAddress("weight_leptonSF_MU_SF_ID_SYST_LOWPT_UP", &weight_leptonSF_MU_SF_ID_SYST_LOWPT_UP, &b_weight_leptonSF_MU_SF_ID_SYST_LOWPT_UP);
      fChain->SetBranchAddress("weight_leptonSF_MU_SF_ID_SYST_LOWPT_DOWN", &weight_leptonSF_MU_SF_ID_SYST_LOWPT_DOWN, &b_weight_leptonSF_MU_SF_ID_SYST_LOWPT_DOWN);
      fChain->SetBranchAddress("weight_leptonSF_MU_SF_Isol_STAT_UP", &weight_leptonSF_MU_SF_Isol_STAT_UP, &b_weight_leptonSF_MU_SF_Isol_STAT_UP);
      fChain->SetBranchAddress("weight_leptonSF_MU_SF_Isol_STAT_DOWN", &weight_leptonSF_MU_SF_Isol_STAT_DOWN, &b_weight_leptonSF_MU_SF_Isol_STAT_DOWN);
      fChain->SetBranchAddress("weight_leptonSF_MU_SF_Isol_SYST_UP", &weight_leptonSF_MU_SF_Isol_SYST_UP, &b_weight_leptonSF_MU_SF_Isol_SYST_UP);
      fChain->SetBranchAddress("weight_leptonSF_MU_SF_Isol_SYST_DOWN", &weight_leptonSF_MU_SF_Isol_SYST_DOWN, &b_weight_leptonSF_MU_SF_Isol_SYST_DOWN);
      fChain->SetBranchAddress("weight_leptonSF_MU_SF_TTVA_STAT_UP", &weight_leptonSF_MU_SF_TTVA_STAT_UP, &b_weight_leptonSF_MU_SF_TTVA_STAT_UP);
      fChain->SetBranchAddress("weight_leptonSF_MU_SF_TTVA_STAT_DOWN", &weight_leptonSF_MU_SF_TTVA_STAT_DOWN, &b_weight_leptonSF_MU_SF_TTVA_STAT_DOWN);
      fChain->SetBranchAddress("weight_leptonSF_MU_SF_TTVA_SYST_UP", &weight_leptonSF_MU_SF_TTVA_SYST_UP, &b_weight_leptonSF_MU_SF_TTVA_SYST_UP);
      fChain->SetBranchAddress("weight_leptonSF_MU_SF_TTVA_SYST_DOWN", &weight_leptonSF_MU_SF_TTVA_SYST_DOWN, &b_weight_leptonSF_MU_SF_TTVA_SYST_DOWN);
      fChain->SetBranchAddress("weight_globalLeptonTriggerSF_EL_Trigger_UP", &weight_globalLeptonTriggerSF_EL_Trigger_UP, &b_weight_globalLeptonTriggerSF_EL_Trigger_UP);
      fChain->SetBranchAddress("weight_globalLeptonTriggerSF_EL_Trigger_DOWN", &weight_globalLeptonTriggerSF_EL_Trigger_DOWN, &b_weight_globalLeptonTriggerSF_EL_Trigger_DOWN);
      fChain->SetBranchAddress("weight_globalLeptonTriggerSF_MU_Trigger_STAT_UP", &weight_globalLeptonTriggerSF_MU_Trigger_STAT_UP, &b_weight_globalLeptonTriggerSF_MU_Trigger_STAT_UP);
      fChain->SetBranchAddress("weight_globalLeptonTriggerSF_MU_Trigger_STAT_DOWN", &weight_globalLeptonTriggerSF_MU_Trigger_STAT_DOWN, &b_weight_globalLeptonTriggerSF_MU_Trigger_STAT_DOWN);
      fChain->SetBranchAddress("weight_globalLeptonTriggerSF_MU_Trigger_SYST_UP", &weight_globalLeptonTriggerSF_MU_Trigger_SYST_UP, &b_weight_globalLeptonTriggerSF_MU_Trigger_SYST_UP);
      fChain->SetBranchAddress("weight_globalLeptonTriggerSF_MU_Trigger_SYST_DOWN", &weight_globalLeptonTriggerSF_MU_Trigger_SYST_DOWN, &b_weight_globalLeptonTriggerSF_MU_Trigger_SYST_DOWN);
      fChain->SetBranchAddress("weight_oldTriggerSF_EL_Trigger_UP", &weight_oldTriggerSF_EL_Trigger_UP, &b_weight_oldTriggerSF_EL_Trigger_UP);
      fChain->SetBranchAddress("weight_oldTriggerSF_EL_Trigger_DOWN", &weight_oldTriggerSF_EL_Trigger_DOWN, &b_weight_oldTriggerSF_EL_Trigger_DOWN);
      fChain->SetBranchAddress("weight_oldTriggerSF_MU_Trigger_STAT_UP", &weight_oldTriggerSF_MU_Trigger_STAT_UP, &b_weight_oldTriggerSF_MU_Trigger_STAT_UP);
      fChain->SetBranchAddress("weight_oldTriggerSF_MU_Trigger_STAT_DOWN", &weight_oldTriggerSF_MU_Trigger_STAT_DOWN, &b_weight_oldTriggerSF_MU_Trigger_STAT_DOWN);
      fChain->SetBranchAddress("weight_oldTriggerSF_MU_Trigger_SYST_UP", &weight_oldTriggerSF_MU_Trigger_SYST_UP, &b_weight_oldTriggerSF_MU_Trigger_SYST_UP);
      fChain->SetBranchAddress("weight_oldTriggerSF_MU_Trigger_SYST_DOWN", &weight_oldTriggerSF_MU_Trigger_SYST_DOWN, &b_weight_oldTriggerSF_MU_Trigger_SYST_DOWN);
      fChain->SetBranchAddress("weight_jvt_UP", &weight_jvt_UP, &b_weight_jvt_UP);
      fChain->SetBranchAddress("weight_jvt_DOWN", &weight_jvt_DOWN, &b_weight_jvt_DOWN);
      fChain->SetBranchAddress("weight_bTagSF_DL1r_Continuous_eigenvars_B_up", &weight_bTagSF_DL1r_Continuous_eigenvars_B_up, &b_weight_bTagSF_DL1r_Continuous_eigenvars_B_up);
      fChain->SetBranchAddress("weight_bTagSF_DL1r_Continuous_eigenvars_C_up", &weight_bTagSF_DL1r_Continuous_eigenvars_C_up, &b_weight_bTagSF_DL1r_Continuous_eigenvars_C_up);
      fChain->SetBranchAddress("weight_bTagSF_DL1r_Continuous_eigenvars_Light_up", &weight_bTagSF_DL1r_Continuous_eigenvars_Light_up, &b_weight_bTagSF_DL1r_Continuous_eigenvars_Light_up);
      fChain->SetBranchAddress("weight_bTagSF_DL1r_Continuous_eigenvars_B_down", &weight_bTagSF_DL1r_Continuous_eigenvars_B_down, &b_weight_bTagSF_DL1r_Continuous_eigenvars_B_down);
      fChain->SetBranchAddress("weight_bTagSF_DL1r_Continuous_eigenvars_C_down", &weight_bTagSF_DL1r_Continuous_eigenvars_C_down, &b_weight_bTagSF_DL1r_Continuous_eigenvars_C_down);
      fChain->SetBranchAddress("weight_bTagSF_DL1r_Continuous_eigenvars_Light_down", &weight_bTagSF_DL1r_Continuous_eigenvars_Light_down, &b_weight_bTagSF_DL1r_Continuous_eigenvars_Light_down);
      fChain->SetBranchAddress("weight_trackjet_bTagSF_DL1r_Continuous_eigenvars_B_up", &weight_trackjet_bTagSF_DL1r_Continuous_eigenvars_B_up, &b_weight_trackjet_bTagSF_DL1r_Continuous_eigenvars_B_up);
      fChain->SetBranchAddress("weight_trackjet_bTagSF_DL1r_Continuous_eigenvars_C_up", &weight_trackjet_bTagSF_DL1r_Continuous_eigenvars_C_up, &b_weight_trackjet_bTagSF_DL1r_Continuous_eigenvars_C_up);
      fChain->SetBranchAddress("weight_trackjet_bTagSF_DL1r_Continuous_eigenvars_Light_up", &weight_trackjet_bTagSF_DL1r_Continuous_eigenvars_Light_up, &b_weight_trackjet_bTagSF_DL1r_Continuous_eigenvars_Light_up);
      fChain->SetBranchAddress("weight_trackjet_bTagSF_DL1r_Continuous_eigenvars_B_down", &weight_trackjet_bTagSF_DL1r_Continuous_eigenvars_B_down, &b_weight_trackjet_bTagSF_DL1r_Continuous_eigenvars_B_down);
      fChain->SetBranchAddress("weight_trackjet_bTagSF_DL1r_Continuous_eigenvars_C_down", &weight_trackjet_bTagSF_DL1r_Continuous_eigenvars_C_down, &b_weight_trackjet_bTagSF_DL1r_Continuous_eigenvars_C_down);
      fChain->SetBranchAddress("weight_trackjet_bTagSF_DL1r_Continuous_eigenvars_Light_down", &weight_trackjet_bTagSF_DL1r_Continuous_eigenvars_Light_down, &b_weight_trackjet_bTagSF_DL1r_Continuous_eigenvars_Light_down);
   }
   fChain->SetBranchAddress("eventNumber", &eventNumber, &b_eventNumber);
   fChain->SetBranchAddress("runNumber", &runNumber, &b_runNumber);
   if (weights_included){
      fChain->SetBranchAddress("randomRunNumber", &randomRunNumber, &b_randomRunNumber);
   }
   fChain->SetBranchAddress("mcChannelNumber", &mcChannelNumber, &b_mcChannelNumber);
   fChain->SetBranchAddress("mu", &mu, &b_mu);
   fChain->SetBranchAddress("mu_actual", &mu_actual, &b_mu_actual);
   fChain->SetBranchAddress("el_pt", &el_pt, &b_el_pt);
   fChain->SetBranchAddress("el_eta", &el_eta, &b_el_eta);
   fChain->SetBranchAddress("el_cl_eta", &el_cl_eta, &b_el_cl_eta);
   fChain->SetBranchAddress("el_phi", &el_phi, &b_el_phi);
   fChain->SetBranchAddress("el_e", &el_e, &b_el_e);
   fChain->SetBranchAddress("el_charge", &el_charge, &b_el_charge);
   fChain->SetBranchAddress("el_topoetcone20", &el_topoetcone20, &b_el_topoetcone20);
   fChain->SetBranchAddress("el_ptvarcone20", &el_ptvarcone20, &b_el_ptvarcone20);
   fChain->SetBranchAddress("el_isTight", &el_isTight, &b_el_isTight);
   fChain->SetBranchAddress("el_CF", &el_CF, &b_el_CF);
   fChain->SetBranchAddress("el_d0sig", &el_d0sig, &b_el_d0sig);
   fChain->SetBranchAddress("el_delta_z0_sintheta", &el_delta_z0_sintheta, &b_el_delta_z0_sintheta);
   fChain->SetBranchAddress("mu_pt", &mu_pt, &b_mu_pt);
   fChain->SetBranchAddress("mu_eta", &mu_eta, &b_mu_eta);
   fChain->SetBranchAddress("mu_phi", &mu_phi, &b_mu_phi);
   fChain->SetBranchAddress("mu_e", &mu_e, &b_mu_e);
   fChain->SetBranchAddress("mu_charge", &mu_charge, &b_mu_charge);
   fChain->SetBranchAddress("mu_topoetcone20", &mu_topoetcone20, &b_mu_topoetcone20);
   fChain->SetBranchAddress("mu_ptvarcone30", &mu_ptvarcone30, &b_mu_ptvarcone30);
   fChain->SetBranchAddress("mu_isTight", &mu_isTight, &b_mu_isTight);
   fChain->SetBranchAddress("mu_d0sig", &mu_d0sig, &b_mu_d0sig);
   fChain->SetBranchAddress("mu_delta_z0_sintheta", &mu_delta_z0_sintheta, &b_mu_delta_z0_sintheta);
   fChain->SetBranchAddress("jet_pt", &jet_pt, &b_jet_pt);
   fChain->SetBranchAddress("jet_eta", &jet_eta, &b_jet_eta);
   fChain->SetBranchAddress("jet_phi", &jet_phi, &b_jet_phi);
   fChain->SetBranchAddress("jet_e", &jet_e, &b_jet_e);
   fChain->SetBranchAddress("jet_jvt", &jet_jvt, &b_jet_jvt);
   if (weights_included){
      fChain->SetBranchAddress("jet_truthflav", &jet_truthflav, &b_jet_truthflav);
      fChain->SetBranchAddress("jet_truthflavExtended", &jet_truthflavExtended, &b_jet_truthflavExtended);
   }
   fChain->SetBranchAddress("jet_isbtagged_DL1r_60", &jet_isbtagged_DL1r_60, &b_jet_isbtagged_DL1r_60);
   fChain->SetBranchAddress("jet_isbtagged_DL1r_70", &jet_isbtagged_DL1r_70, &b_jet_isbtagged_DL1r_70);
   fChain->SetBranchAddress("jet_isbtagged_DL1r_77", &jet_isbtagged_DL1r_77, &b_jet_isbtagged_DL1r_77);
   fChain->SetBranchAddress("jet_isbtagged_DL1r_85", &jet_isbtagged_DL1r_85, &b_jet_isbtagged_DL1r_85);
   fChain->SetBranchAddress("jet_tagWeightBin_DL1r_Continuous", &jet_tagWeightBin_DL1r_Continuous, &b_jet_tagWeightBin_DL1r_Continuous);
   fChain->SetBranchAddress("jet_DL1r", &jet_DL1r, &b_jet_DL1r);
   fChain->SetBranchAddress("ljet_pt", &ljet_pt, &b_ljet_pt);
   fChain->SetBranchAddress("ljet_eta", &ljet_eta, &b_ljet_eta);
   fChain->SetBranchAddress("ljet_phi", &ljet_phi, &b_ljet_phi);
   fChain->SetBranchAddress("ljet_e", &ljet_e, &b_ljet_e);
   fChain->SetBranchAddress("ljet_m", &ljet_m, &b_ljet_m);
   if (weights_included){
      fChain->SetBranchAddress("ljet_truthLabel", &ljet_truthLabel, &b_ljet_truthLabel);
   }
   fChain->SetBranchAddress("tjet_pt", &tjet_pt, &b_tjet_pt);
   fChain->SetBranchAddress("tjet_eta", &tjet_eta, &b_tjet_eta);
   fChain->SetBranchAddress("tjet_phi", &tjet_phi, &b_tjet_phi);
   fChain->SetBranchAddress("tjet_e", &tjet_e, &b_tjet_e);
   fChain->SetBranchAddress("tjet_tagWeightBin_DL1r_Continuous", &tjet_tagWeightBin_DL1r_Continuous, &b_tjet_tagWeightBin_DL1r_Continuous);
   fChain->SetBranchAddress("tjet_DL1r", &tjet_DL1r, &b_tjet_DL1r);
   fChain->SetBranchAddress("met_met", &met_met, &b_met_met);
   fChain->SetBranchAddress("met_phi", &met_phi, &b_met_phi);
   //fChain->SetBranchAddress("ejets_2018_DL1r", &ejets_2018_DL1r, &b_ejets_2018_DL1r);
   //fChain->SetBranchAddress("mujets_2018_DL1r", &mujets_2018_DL1r, &b_mujets_2018_DL1r);
   //fChain->SetBranchAddress("boosted_ljets_ejets_2018_DL1r", &boosted_ljets_ejets_2018_DL1r, &b_boosted_ljets_ejets_2018_DL1r);
   //fChain->SetBranchAddress("boosted_ljets_mujets_2018_DL1r", &boosted_ljets_mujets_2018_DL1r, &b_boosted_ljets_mujets_2018_DL1r);
   if (weights_included){
      fChain->SetBranchAddress("HLT_mu26_ivarmedium", &HLT_mu26_ivarmedium, &b_HLT_mu26_ivarmedium);
   }
   fChain->SetBranchAddress("HLT_mu50", &HLT_mu50, &b_HLT_mu50);
   if (weights_included){
      fChain->SetBranchAddress("HLT_e60_lhmedium_nod0", &HLT_e60_lhmedium_nod0, &b_HLT_e60_lhmedium_nod0);
      fChain->SetBranchAddress("HLT_e140_lhloose_nod0", &HLT_e140_lhloose_nod0, &b_HLT_e140_lhloose_nod0);
      fChain->SetBranchAddress("HLT_e26_lhtight_nod0_ivarloose", &HLT_e26_lhtight_nod0_ivarloose, &b_HLT_e26_lhtight_nod0_ivarloose);
   }
   fChain->SetBranchAddress("ljet_C2", &ljet_C2, &b_ljet_C2);
   fChain->SetBranchAddress("ljet_D2", &ljet_D2, &b_ljet_D2);
   fChain->SetBranchAddress("ljet_Xbb2020v3_Higgs", &ljet_Xbb2020v3_Higgs, &b_ljet_Xbb2020v3_Higgs);
   fChain->SetBranchAddress("ljet_Xbb2020v3_QCD", &ljet_Xbb2020v3_QCD, &b_ljet_Xbb2020v3_QCD);
   fChain->SetBranchAddress("ljet_Xbb2020v3_Top", &ljet_Xbb2020v3_Top, &b_ljet_Xbb2020v3_Top);
   fChain->SetBranchAddress("ljet_muonCorrectedEta", &ljet_muonCorrectedEta, &b_ljet_muonCorrectedEta);
   fChain->SetBranchAddress("ljet_muonCorrectedMass", &ljet_muonCorrectedMass, &b_ljet_muonCorrectedMass);
   fChain->SetBranchAddress("ljet_muonCorrectedPhi", &ljet_muonCorrectedPhi, &b_ljet_muonCorrectedPhi);
   fChain->SetBranchAddress("ljet_muonCorrectedPt", &ljet_muonCorrectedPt, &b_ljet_muonCorrectedPt);
   if (weights_included){
      fChain->SetBranchAddress("HF_Classification", &HF_Classification, &b_HF_Classification);
      fChain->SetBranchAddress("HF_ClassificationGhost", &HF_ClassificationGhost, &b_HF_ClassificationGhost);
      fChain->SetBranchAddress("HF_SimpleClassification", &HF_SimpleClassification, &b_HF_SimpleClassification);
      fChain->SetBranchAddress("HF_SimpleClassificationGhost", &HF_SimpleClassificationGhost, &b_HF_SimpleClassificationGhost);
      fChain->SetBranchAddress("TopHeavyFlavorFilterFlag", &TopHeavyFlavorFilterFlag, &b_TopHeavyFlavorFilterFlag);
   }
   fChain->SetBranchAddress("nBTagsTrackJets_DL1r_60", &nBTagsTrackJets_DL1r_60, &b_nBTagsTrackJets_DL1r_60);
   fChain->SetBranchAddress("nBTagsTrackJets_DL1r_70", &nBTagsTrackJets_DL1r_70, &b_nBTagsTrackJets_DL1r_70);
   fChain->SetBranchAddress("nBTagsTrackJets_DL1r_77", &nBTagsTrackJets_DL1r_77, &b_nBTagsTrackJets_DL1r_77);
   fChain->SetBranchAddress("nBTagsTrackJets_DL1r_85", &nBTagsTrackJets_DL1r_85, &b_nBTagsTrackJets_DL1r_85);
   fChain->SetBranchAddress("nBTags_DL1r_60", &nBTags_DL1r_60, &b_nBTags_DL1r_60);
   fChain->SetBranchAddress("nBTags_DL1r_70", &nBTags_DL1r_70, &b_nBTags_DL1r_70);
   fChain->SetBranchAddress("nBTags_DL1r_77", &nBTags_DL1r_77, &b_nBTags_DL1r_77);
   fChain->SetBranchAddress("nBTags_DL1r_85", &nBTags_DL1r_85, &b_nBTags_DL1r_85);
   fChain->SetBranchAddress("nElectrons", &nElectrons, &b_nElectrons);
   fChain->SetBranchAddress("nJets", &nJets, &b_nJets);
   fChain->SetBranchAddress("nLJets", &nLJets, &b_nLJets);
   fChain->SetBranchAddress("nLJets_matched", &nLJets_matched, &b_nLJets_matched);
   fChain->SetBranchAddress("nMuons", &nMuons, &b_nMuons);
   if (weights_included){
      fChain->SetBranchAddress("nPDFFlavor", &nPDFFlavor, &b_nPDFFlavor);
   }
   fChain->SetBranchAddress("nPrimaryVtx", &nPrimaryVtx, &b_nPrimaryVtx);
   fChain->SetBranchAddress("nTaus", &nTaus, &b_nTaus);
   fChain->SetBranchAddress("Aplanarity_bjets_77", &Aplanarity_bjets_77, &b_Aplanarity_bjets_77);
   fChain->SetBranchAddress("Aplanarity_bjets_85", &Aplanarity_bjets_85, &b_Aplanarity_bjets_85);
   fChain->SetBranchAddress("Aplanarity_bjets_Sort4", &Aplanarity_bjets_Sort4, &b_Aplanarity_bjets_Sort4);
   fChain->SetBranchAddress("Aplanarity_jets", &Aplanarity_jets, &b_Aplanarity_jets);
   fChain->SetBranchAddress("Centrality_all", &Centrality_all, &b_Centrality_all);
   fChain->SetBranchAddress("H0_all", &H0_all, &b_H0_all);
   fChain->SetBranchAddress("H1_all", &H1_all, &b_H1_all);
   fChain->SetBranchAddress("H2_jets", &H2_jets, &b_H2_jets);
   fChain->SetBranchAddress("H4_all", &H4_all, &b_H4_all);
   fChain->SetBranchAddress("HT_all", &HT_all, &b_HT_all);
   fChain->SetBranchAddress("HT_jets", &HT_jets, &b_HT_jets);
   fChain->SetBranchAddress("Mbb_HiggsMass_77", &Mbb_HiggsMass_77, &b_Mbb_HiggsMass_77);
   fChain->SetBranchAddress("Mbb_HiggsMass_85", &Mbb_HiggsMass_85, &b_Mbb_HiggsMass_85);
   fChain->SetBranchAddress("Mbb_HiggsMass_Sort4", &Mbb_HiggsMass_Sort4, &b_Mbb_HiggsMass_Sort4);
   fChain->SetBranchAddress("Mbb_MaxM_77", &Mbb_MaxM_77, &b_Mbb_MaxM_77);
   fChain->SetBranchAddress("Mbb_MaxM_85", &Mbb_MaxM_85, &b_Mbb_MaxM_85);
   fChain->SetBranchAddress("Mbb_MaxM_Sort4", &Mbb_MaxM_Sort4, &b_Mbb_MaxM_Sort4);
   fChain->SetBranchAddress("Mbb_MaxPt_77", &Mbb_MaxPt_77, &b_Mbb_MaxPt_77);
   fChain->SetBranchAddress("Mbb_MaxPt_85", &Mbb_MaxPt_85, &b_Mbb_MaxPt_85);
   fChain->SetBranchAddress("Mbb_MaxPt_Sort4", &Mbb_MaxPt_Sort4, &b_Mbb_MaxPt_Sort4);
   fChain->SetBranchAddress("Mbb_MinM_77", &Mbb_MinM_77, &b_Mbb_MinM_77);
   fChain->SetBranchAddress("Mbb_MinM_85", &Mbb_MinM_85, &b_Mbb_MinM_85);
   fChain->SetBranchAddress("Mbb_MinM_Sort4", &Mbb_MinM_Sort4, &b_Mbb_MinM_Sort4);
   fChain->SetBranchAddress("Mbb_MindR_77", &Mbb_MindR_77, &b_Mbb_MindR_77);
   fChain->SetBranchAddress("Mbb_MindR_85", &Mbb_MindR_85, &b_Mbb_MindR_85);
   fChain->SetBranchAddress("Mbb_MindR_Sort4", &Mbb_MindR_Sort4, &b_Mbb_MindR_Sort4);
   fChain->SetBranchAddress("Mbj_MaxPt_77", &Mbj_MaxPt_77, &b_Mbj_MaxPt_77);
   fChain->SetBranchAddress("Mbj_MaxPt_85", &Mbj_MaxPt_85, &b_Mbj_MaxPt_85);
   fChain->SetBranchAddress("Mbj_MaxPt_Sort4", &Mbj_MaxPt_Sort4, &b_Mbj_MaxPt_Sort4);
   fChain->SetBranchAddress("Mbj_MindR_77", &Mbj_MindR_77, &b_Mbj_MindR_77);
   fChain->SetBranchAddress("Mbj_MindR_85", &Mbj_MindR_85, &b_Mbj_MindR_85);
   fChain->SetBranchAddress("Mbj_MindR_Sort4", &Mbj_MindR_Sort4, &b_Mbj_MindR_Sort4);
   fChain->SetBranchAddress("Mbj_Wmass_77", &Mbj_Wmass_77, &b_Mbj_Wmass_77);
   fChain->SetBranchAddress("Mbj_Wmass_85", &Mbj_Wmass_85, &b_Mbj_Wmass_85);
   fChain->SetBranchAddress("Mbj_Wmass_Sort4", &Mbj_Wmass_Sort4, &b_Mbj_Wmass_Sort4);
   fChain->SetBranchAddress("Mjj_HiggsMass", &Mjj_HiggsMass, &b_Mjj_HiggsMass);
   fChain->SetBranchAddress("Mjj_MaxPt", &Mjj_MaxPt, &b_Mjj_MaxPt);
   fChain->SetBranchAddress("Mjj_MinM", &Mjj_MinM, &b_Mjj_MinM);
   fChain->SetBranchAddress("Mjj_MindR", &Mjj_MindR, &b_Mjj_MindR);
   fChain->SetBranchAddress("Mjjj_MaxPt", &Mjjj_MaxPt, &b_Mjjj_MaxPt);
   fChain->SetBranchAddress("Muu_MindR_77", &Muu_MindR_77, &b_Muu_MindR_77);
   fChain->SetBranchAddress("Muu_MindR_85", &Muu_MindR_85, &b_Muu_MindR_85);
   fChain->SetBranchAddress("Muu_MindR_Sort4", &Muu_MindR_Sort4, &b_Muu_MindR_Sort4);
   fChain->SetBranchAddress("dRbb_MaxM_77", &dRbb_MaxM_77, &b_dRbb_MaxM_77);
   fChain->SetBranchAddress("dRbb_MaxM_85", &dRbb_MaxM_85, &b_dRbb_MaxM_85);
   fChain->SetBranchAddress("dRbb_MaxM_Sort4", &dRbb_MaxM_Sort4, &b_dRbb_MaxM_Sort4);
   fChain->SetBranchAddress("dRbb_MaxPt_77", &dRbb_MaxPt_77, &b_dRbb_MaxPt_77);
   fChain->SetBranchAddress("dRbb_MaxPt_85", &dRbb_MaxPt_85, &b_dRbb_MaxPt_85);
   fChain->SetBranchAddress("dRbb_MaxPt_Sort4", &dRbb_MaxPt_Sort4, &b_dRbb_MaxPt_Sort4);
   fChain->SetBranchAddress("dRbb_MindR_77", &dRbb_MindR_77, &b_dRbb_MindR_77);
   fChain->SetBranchAddress("dRbb_MindR_85", &dRbb_MindR_85, &b_dRbb_MindR_85);
   fChain->SetBranchAddress("dRbb_MindR_Sort4", &dRbb_MindR_Sort4, &b_dRbb_MindR_Sort4);
   fChain->SetBranchAddress("dRbb_avg_77", &dRbb_avg_77, &b_dRbb_avg_77);
   fChain->SetBranchAddress("dRbb_avg_85", &dRbb_avg_85, &b_dRbb_avg_85);
   fChain->SetBranchAddress("dRbb_avg_Sort4", &dRbb_avg_Sort4, &b_dRbb_avg_Sort4);
   fChain->SetBranchAddress("dRbj_Wmass_77", &dRbj_Wmass_77, &b_dRbj_Wmass_77);
   fChain->SetBranchAddress("dRbj_Wmass_85", &dRbj_Wmass_85, &b_dRbj_Wmass_85);
   fChain->SetBranchAddress("dRbj_Wmass_Sort4", &dRbj_Wmass_Sort4, &b_dRbj_Wmass_Sort4);
   fChain->SetBranchAddress("dRlepbb_MindR_77", &dRlepbb_MindR_77, &b_dRlepbb_MindR_77);
   fChain->SetBranchAddress("dRlepbb_MindR_85", &dRlepbb_MindR_85, &b_dRlepbb_MindR_85);
   fChain->SetBranchAddress("dRlepbb_MindR_Sort4", &dRlepbb_MindR_Sort4, &b_dRlepbb_MindR_Sort4);
   fChain->SetBranchAddress("dRlj_MindR", &dRlj_MindR, &b_dRlj_MindR);
   fChain->SetBranchAddress("dRuu_MindR_77", &dRuu_MindR_77, &b_dRuu_MindR_77);
   fChain->SetBranchAddress("dRuu_MindR_85", &dRuu_MindR_85, &b_dRuu_MindR_85);
   fChain->SetBranchAddress("dRuu_MindR_Sort4", &dRuu_MindR_Sort4, &b_dRuu_MindR_Sort4);
   fChain->SetBranchAddress("pT_jet3", &pT_jet3, &b_pT_jet3);
   fChain->SetBranchAddress("pT_jet5", &pT_jet5, &b_pT_jet5);
   fChain->SetBranchAddress("pTbb_MindR_77", &pTbb_MindR_77, &b_pTbb_MindR_77);
   fChain->SetBranchAddress("pTbb_MindR_85", &pTbb_MindR_85, &b_pTbb_MindR_85);
   fChain->SetBranchAddress("pTbb_MindR_Sort4", &pTbb_MindR_Sort4, &b_pTbb_MindR_Sort4);
   fChain->SetBranchAddress("pTuu_MindR_77", &pTuu_MindR_77, &b_pTuu_MindR_77);
   fChain->SetBranchAddress("pTuu_MindR_85", &pTuu_MindR_85, &b_pTuu_MindR_85);
   fChain->SetBranchAddress("pTuu_MindR_Sort4", &pTuu_MindR_Sort4, &b_pTuu_MindR_Sort4);
   if (weights_included){
      fChain->SetBranchAddress("truth_pt", &truth_pt, &b_truth_pt);
      fChain->SetBranchAddress("truth_eta", &truth_eta, &b_truth_eta);
      fChain->SetBranchAddress("truth_phi", &truth_phi, &b_truth_phi);
      fChain->SetBranchAddress("truth_m", &truth_m, &b_truth_m);
      fChain->SetBranchAddress("truth_pdgid", &truth_pdgid, &b_truth_pdgid);
      fChain->SetBranchAddress("truth_status", &truth_status, &b_truth_status);
      fChain->SetBranchAddress("truth_barcode", &truth_barcode, &b_truth_barcode);
      fChain->SetBranchAddress("truth_tthbb_info", &truth_tthbb_info, &b_truth_tthbb_info);
      fChain->SetBranchAddress("truth_jet_pt", &truth_jet_pt, &b_truth_jet_pt);
      fChain->SetBranchAddress("truth_jet_eta", &truth_jet_eta, &b_truth_jet_eta);
      fChain->SetBranchAddress("truth_jet_phi", &truth_jet_phi, &b_truth_jet_phi);
      fChain->SetBranchAddress("truth_jet_m", &truth_jet_m, &b_truth_jet_m);
   }
   Notify();
}

Bool_t EventLoop::Notify()
{
   // The Notify() function is called when a new file is opened. This
   // can be either for a new TTree in a TChain or when when a new TTree
   // is started when using PROOF. It is normally not necessary to make changes
   // to the generated code, but the routine can be extended by the
   // user if needed. The return value is currently not used.

   return kTRUE;
}

void EventLoop::Show(Long64_t entry)
{
// Print contents of entry.
// If entry is not specified, print current entry
   if (!fChain) return;
   fChain->Show(entry);
}
Int_t EventLoop::Cut(Long64_t entry)
{
// This function may be called from Loop.
// returns  1 if entry is accepted.
// returns -1 otherwise.
   return 1;
}
#endif // #ifdef EventLoop_cxx
