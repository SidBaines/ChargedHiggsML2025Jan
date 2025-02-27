#include <TFile.h>
#include <TTree.h>
#include <TString.h>
#include <string> 
#include <fstream>
#include <iostream>
#include <vector>
int main(int argc, char **argv)
{
    TFile *myfile = TFile::Open(argv[1]);
    TTree *mytree = (TTree*)myfile->Get(argv[2]);
    int num = mytree->GetEntries();
    //delete mytree;
    myfile->Close();
    delete myfile;
    std::cout << num << std::endl;
    return 0;
}