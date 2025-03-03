#define EventLoop_cxx
#include "EventLoop.h"
#include <TH2.h>
#include <TStyle.h>
#include <TCanvas.h>
#include <tuple>
/*
Comments from Dominik
        //USE TRACK JETS FOR OUR B TAGGING
        //Mainly want to condier btags from the top quark decay, or the btags for the jets within the H+ decay
        //Exclusively outside the higgs or W
*/


int EventLoop::LowLevel_ClassifyDecayType_OLD(){
    const double W_MASS = 80.379e3; // GeV to MeV
    const double W_MASS_WINDOW =5.0e3; // GeV to MeV
    // Check if truth information exists
    if (truth_pt->empty() || truth_eta->empty() || truth_phi->empty() || 
        truth_m->empty() || truth_pdgid->empty()) {
        return -3;
    }

    // Create vectors to store particle information
    std::vector<TLorentzVector> tparticleP4;
    std::vector<int> pdgIds;
    // Fill vectors with truth information
    for (size_t i = 0; i < truth_pt->size(); i++) {
        TLorentzVector p4;
        p4.SetPtEtaPhiM(truth_pt->at(i), truth_eta->at(i), 
                        truth_phi->at(i), truth_m->at(i));
        tparticleP4.push_back(p4);
        pdgIds.push_back(truth_pdgid->at(i));
    }

    if (false){ // Testing 15/01/25
        for (size_t i = 0; i < pdgIds.size(); i++) {
            if (pdgIds[i] == 24) {
                std::cout << "TRUTH W+ : (" << tparticleP4[i].Pt() << ", " << tparticleP4[i].Eta() << ", " << tparticleP4[i].Phi() << ", " << tparticleP4[i].M() << std::endl;
            }
            if (pdgIds[i] == -24) {
                std::cout << "TRUTH W- : (" << tparticleP4[i].Pt() << ", " << tparticleP4[i].Eta() << ", " << tparticleP4[i].Phi() << ", " << tparticleP4[i].M() << std::endl;
            }
        }
    }

    // Find charged Higgs (pdgId = ±37)
    int chHiggsIdx = -1;
    TLorentzVector chHiggsP4;
    for (size_t i = 0; i < pdgIds.size(); i++) {
        if (abs(pdgIds[i]) == 37) {
            chHiggsIdx = i;
            chHiggsP4 = tparticleP4[i];
            break;
        }
    }
    if (chHiggsIdx == -1) return -4;
    // Find SM Higgs (pdgId = 25)
    int smHiggsIdx = -1;
    TLorentzVector smHiggsP4;
    for (size_t i = 0; i < pdgIds.size(); i++) {
        if (pdgIds[i] == 25) {
            smHiggsIdx = i;
            smHiggsP4 = tparticleP4[i];
            break;
        }
    }
    if (smHiggsIdx == -1) return -5;
    ll_truth_Higgs_old.SetPtEtaPhiM(smHiggsP4.Pt(), smHiggsP4.Eta(), smHiggsP4.Phi(), smHiggsP4.M());
    // Check for b-quark pair from Higgs decay
    bool foundBPair = false;
    int bCount = 0;
    for (int pdgId : pdgIds) {
        if (abs(pdgId) == 5) bCount++;
    }
    if (bCount >= 2) foundBPair = true;
    if (!foundBPair) return -6;
    // truth_Higgs_reco = ???;
    // Find W boson that best reconstructs charged Higgs when combined with SM Higgs
    double bestDeltaM = 1e12;
    int bestWIdx = -1;
    TLorentzVector bestWP4;
    for (size_t i = 0; i < pdgIds.size(); i++) {
        // if (abs(pdgIds[i]) == 24) {
        if (pdgIds[i] == 24) {
            double deltaM = abs((tparticleP4[i] + smHiggsP4).M() - chHiggsP4.M());
            if (deltaM < bestDeltaM) {
                bestDeltaM = deltaM;
                bestWIdx = i;
                bestWP4 = tparticleP4[i];
            }
        }
    }
    if (bestWIdx == -1) return -7;
    ll_truth_W_old.SetPtEtaPhiM(bestWP4.Pt(), bestWP4.Eta(), bestWP4.Phi(), bestWP4.M());
    int decay_type = 0;
    int lepton_idx = -1;
    if (false){ // Old version, doesn't account for 2 lepton case
        // First find the lepton
        for(size_t i = 0; i < truth_pdgid->size(); i++) {
            int pdgid = (*truth_pdgid)[i];
            // Check for electron (11) or muon (13)
            if(abs(pdgid) == 11 || abs(pdgid) == 13) {
                lepton_idx = i;
                // Set decay type based on charge (negative pdgid = positive particle)
                decay_type = (pdgid > 0) ? 2 : 1;
                break;
            }
        }
    } else{ // New version, checks for which truth lepton most closely matches the event-level lepton
        // First find the lepton
        float min_delta_R = -1;
        for(size_t i = 0; i < truth_pdgid->size(); i++) {
            int pdgid = (*truth_pdgid)[i];
            // Check for electron (11) or muon (13)
            if(abs(pdgid) == 11 || abs(pdgid) == 13) {
                if ((min_delta_R==-1) || (particles.at(0).p4.DeltaR(tparticleP4.at(i))) < min_delta_R){
                    lepton_idx = i;
                    min_delta_R = (particles.at(0).p4.DeltaR(tparticleP4.at(i)));
                    // Set decay type based on charge (negative pdgid = positive particle)
                    decay_type = (pdgid > 0) ? 2 : 1;
                }
            }
        }
    }
    if(lepton_idx == -1) return -8; // No lepton found
    // Create lepton TLorentzVector
    TLorentzVector lepton;
    lepton.SetPtEtaPhiM((*truth_pt)[lepton_idx],
                        (*truth_eta)[lepton_idx],
                        (*truth_phi)[lepton_idx],
                        (*truth_m)[lepton_idx]);
    if(decay_type == 1) {
        // Look for corresponding neutrino
        int target_nu_pdgid;
        if(abs((*truth_pdgid)[lepton_idx]) == 11) {
            target_nu_pdgid = 12; // electron neutrino
        } else {
            target_nu_pdgid = 14; // muon neutrino
        }
        
        int nu_idx = -1;
        for(size_t i = 0; i < truth_pdgid->size(); i++) {
            if((*truth_pdgid)[i] == target_nu_pdgid) {
                nu_idx = i;
                break;
            }
        }
        if(nu_idx == -1) return -1; // No matching neutrino found
        // Create neutrino TLorentzVector and sum with lepton
        TLorentzVector neutrino;
        neutrino.SetPtEtaPhiM((*truth_pt)[nu_idx],
                             (*truth_eta)[nu_idx],
                             (*truth_phi)[nu_idx],
                             (*truth_m)[nu_idx]);
        ll_truth_W_reco_old = lepton + neutrino;
    } else if(decay_type == 2) {
        // Look for jet pair with mass close to W mass
        double best_deltaR = 999.0;
        int best_j1_idx = -1;
        int best_j2_idx = -1;
        for(size_t i = 0; i < truth_pdgid->size(); i++) {
            // Skip if not a jet (assuming jets have pdgid < 7)
            if(abs((*truth_pdgid)[i]) >= 6) continue;
            TLorentzVector jet1;
            jet1.SetPtEtaPhiM((*truth_pt)[i],
                             (*truth_eta)[i],
                             (*truth_phi)[i],
                             (*truth_m)[i]);
            for(size_t j = i + 1; j < truth_pdgid->size(); j++) {
                if(abs((*truth_pdgid)[j]) >= 6) continue;
                TLorentzVector jet2;
                jet2.SetPtEtaPhiM((*truth_pt)[j],
                                 (*truth_eta)[j],
                                 (*truth_phi)[j],
                                 (*truth_m)[j]);
                TLorentzVector dijet = jet1 + jet2;
                double mass_diff = abs(dijet.M() - W_MASS);
                if(mass_diff < W_MASS_WINDOW) {
                    double deltaR = dijet.DeltaR(ll_truth_W_old);
                    if(deltaR < best_deltaR) {
                        best_deltaR = deltaR;
                        best_j1_idx = i;
                        best_j2_idx = j;
                    }
                }
            }
        }
        if(best_j1_idx == -1 || best_j2_idx == -1) return -2; // No suitable jet pair found
        // Reconstruct W from best jet pair
        TLorentzVector jet1, jet2;
        jet1.SetPtEtaPhiM((*truth_pt)[best_j1_idx],
                          (*truth_eta)[best_j1_idx],
                          (*truth_phi)[best_j1_idx],
                          (*truth_m)[best_j1_idx]);
        jet2.SetPtEtaPhiM((*truth_pt)[best_j2_idx],
                          (*truth_eta)[best_j2_idx],
                          (*truth_phi)[best_j2_idx],
                          (*truth_m)[best_j2_idx]);
        
        ll_truth_W_reco_old = jet1 + jet2;
    }
    for(size_t i = 0; i < truth_pdgid->size(); i++) {
        if (((*truth_pdgid)[i] == -15) and (decay_type == 1)) return -9; // Tau+ in the event truth-particles so this case is tricky, discount it
        if (((*truth_pdgid)[i] == -15) and (decay_type == 2)) return -10; // Tau+ in the event truth-particles so this case is tricky, discount it
    }
    return decay_type;
}

void EventLoop::Fill_NN_Scores(){
    // Fill the inputs to the NN
	m_inputs_lvbb["LepEnergyFrac_lvbb"] = m_LepEnergyFrac_lvbb;
	m_inputs_lvbb["mTop_lepto"] = m_mTop_lepto;
	m_inputs_lvbb["dR_LH"] = m_deltaR_LH;
	m_inputs_lvbb["DeltaEta_H_WLep"] = m_deltaEta_HWlep;
	m_inputs_lvbb["RWpTM_lvbb"] = ratio_Wpt_mVH_lvbb;
	m_inputs_lvbb["RHpTM_lvbb"] = ratio_Hpt_mVH_lvbb;
	m_inputs_lvbb["DeltaPhi_H_WLep"] = m_deltaPhi_HWlep;

	m_inputs_qqbb["LepEnergyFrac_qqbb"] = m_LepEnergyFrac_qqbb;
	m_inputs_qqbb["RWpTM_qqbb"] = ratio_Wpt_mVH_qqbb;
	m_inputs_qqbb["RHpTM_qqbb"] = ratio_Hpt_mVH_qqbb;
	m_inputs_qqbb["dR_LWhad"] = m_deltaR_LWhad;
	m_inputs_qqbb["dR_LH"] = m_deltaR_LH;
	m_inputs_qqbb["DeltaPhi_H_WHad"] = m_deltaPhi_HWhad;
	m_inputs_qqbb["DeltaEta_H_WHad"] = m_deltaEta_HWhad;

    // Maybe we need to change these ratios to what the NNs are expecting
    m_inputs_lvbb["RWpTM_lvbb"] = m_inputs_lvbb["RWpTM_lvbb"]/1000;
    m_inputs_lvbb["RHpTM_lvbb"] = m_inputs_lvbb["RHpTM_lvbb"]/1000;
    m_inputs_qqbb["RWpTM_qqbb"] = m_inputs_qqbb["RWpTM_qqbb"]/1000;
    m_inputs_qqbb["RHpTM_qqbb"] = m_inputs_qqbb["RHpTM_qqbb"]/1000;
    // Put the dR_LH from qqbb into lvbb, because they have to share this variable from the TBranch and if we don't do this the lvbb one is always 0
    m_inputs_lvbb["dR_LH"] = m_inputs_qqbb["dR_LH"];
    if (false){ // Testing 15/01/25 
        if ((selection_category == 0) || (selection_category == 8) || (selection_category == 10)){
            std::cout << "Selection category: lvbb" << std::endl;
        } else if ((selection_category == 3) || (selection_category == 9)){
            std::cout << "Selection category: qqbb" << std::endl;
        } else {
            std::cout << "Selection category: " << selection_category << std::endl;
        }
        if (false) { // Testing 15/01/25
            
            std::cout << "OLD VERSION RECO W_qq : (" << W_hadronic.Pt() << ", " << W_hadronic.Eta() << ", " << W_hadronic.Phi() << ", " << W_hadronic.M() << std::endl;
            std::cout << "OLD VERSION RECO W_lv : (" << W_leptonic.Pt() << ", " << W_leptonic.Eta() << ", " << W_leptonic.Phi() << ", " << W_leptonic.M() << std::endl;
            std::cout << "mWH_lvbb OLD version: " << m_mVH_lvbb << std::endl;
            std::cout << "mWH_qqbb OLD version: " << m_mVH_qqbb << std::endl;
        }
    }

    bool WriteOnlyCorrectCategory = true; // If true, only fill one of the lvbb/qqbb predictions, depending on the reco channel. The other predictions will be set to -1.
    for (auto kv_pair:nn_map_lvbb){
        if (debugMode) std::cout << "\t" << "Setting NN: " << kv_pair.first << "to -1" << std::endl;
        lvbbNNs[kv_pair.first] = -1;
        if (debugMode) std::cout << "\t" << "Making prediction for NN: " << kv_pair.first << std::endl;
        // Fix a couple of values in terms of GeV vs. MeV for most recent NNs
        if (kv_pair.first.find("Ttbar") != std::string::npos) {
            m_inputs_lvbb["RWpTM_lvbb"] = m_inputs_lvbb["RWpTM_lvbb"]*1000;
            m_inputs_lvbb["RHpTM_lvbb"] = m_inputs_lvbb["RHpTM_lvbb"]*1000;
        }
        if ((WriteOnlyCorrectCategory) && !((selection_category == 0) || (selection_category == 8) || (selection_category == 10))) {
            lvbbNNs[kv_pair.first] = -1;
        }
        else{
            if (kv_pair.first.find("pNN800") != std::string::npos) {
                m_inputs_lvbb["mass"] = 0.8;
                mm_inputs["inputs_node"] = m_inputs_lvbb;
                outputs = kv_pair.second->compute(mm_inputs);
            }
            else if (kv_pair.first.find("pNN900") != std::string::npos) {
                m_inputs_lvbb["mass"] = 0.9;
                mm_inputs["inputs_node"] = m_inputs_lvbb;
                outputs = kv_pair.second->compute(mm_inputs);
            }
            else if (kv_pair.first.find("pNN1000") != std::string::npos) {
                m_inputs_lvbb["mass"] = 1.0;
                mm_inputs["inputs_node"] = m_inputs_lvbb;
                outputs = kv_pair.second->compute(mm_inputs);
            }
            else if (kv_pair.first.find("pNN1200") != std::string::npos) {
                m_inputs_lvbb["mass"] = 1.2;
                mm_inputs["inputs_node"] = m_inputs_lvbb;
                outputs = kv_pair.second->compute(mm_inputs);
            }
            else if (kv_pair.first.find("pNN1400") != std::string::npos) {
                m_inputs_lvbb["mass"] = 1.4;
                mm_inputs["inputs_node"] = m_inputs_lvbb;
                outputs = kv_pair.second->compute(mm_inputs);
            }
            else if (kv_pair.first.find("pNN1600") != std::string::npos) {
                m_inputs_lvbb["mass"] = 1.6;
                mm_inputs["inputs_node"] = m_inputs_lvbb;
                outputs = kv_pair.second->compute(mm_inputs);
            }
            else if (kv_pair.first.find("pNN1800") != std::string::npos) {
                m_inputs_lvbb["mass"] = 1.8;
                mm_inputs["inputs_node"] = m_inputs_lvbb;
                outputs = kv_pair.second->compute(mm_inputs);
            }
            else if (kv_pair.first.find("pNN2000") != std::string::npos) {
                m_inputs_lvbb["mass"] = 2.0;
                mm_inputs["inputs_node"] = m_inputs_lvbb;
                outputs = kv_pair.second->compute(mm_inputs);
            }
            else if (kv_pair.first.find("pNN2500") != std::string::npos) {
                m_inputs_lvbb["mass"] = 2.5;
                mm_inputs["inputs_node"] = m_inputs_lvbb;
                outputs = kv_pair.second->compute(mm_inputs);
            }
            else if (kv_pair.first.find("pNN3000") != std::string::npos) {
                m_inputs_lvbb["mass"] = 3.0;
                mm_inputs["inputs_node"] = m_inputs_lvbb;
                outputs = kv_pair.second->compute(mm_inputs);
            }
            else{
                mm_inputs["inputs_node"] = m_inputs_lvbb;
                if (debugMode) {
                    std::cout << "\t" << "Inputs are: " << std::endl;
                    for (auto &ipt: mm_inputs["inputs_node"]){
                        std::cout << "\t\t" << ipt.first << ": " << ipt.second << std::endl;
                    }
                }
                outputs = kv_pair.second->compute(mm_inputs);
                // if (debugMode) std::cout << "\t" << "Value found is: " << outputs[0].second << std::endl;
            }
            for (auto &opt: outputs){
                lvbbNNs[kv_pair.first] = opt.second;
                if (debugMode) std::cout << "\t" << "Value found is: " << opt.second << std::endl;
            }
        }
        // Fix a couple of values in terms of GeV vs. MeV
        if (kv_pair.first.find("Ttbar") != std::string::npos) {
            m_inputs_lvbb["RWpTM_lvbb"] = m_inputs_lvbb["RWpTM_lvbb"]/1000;
            m_inputs_lvbb["RHpTM_lvbb"] = m_inputs_lvbb["RHpTM_lvbb"]/1000;
        }
    }

    for (auto kv_pair:nn_map_qqbb){
        if (debugMode) std::cout << "\t" << "Setting NN: " << kv_pair.first << "to -1" << std::endl;
        qqbbNNs[kv_pair.first] = -1;
        if (debugMode) std::cout << "\t" << "Making prediction for NN: " << kv_pair.first << std::endl;
        // Fix a couple of values in terms of GeV vs. MeV for most recent NNs
        // bool rescale = true;
        // bool rescale = kv_pair.first.find("Ttbar") != std::string::npos;
        bool rescale = !(kv_pair.first.find("Final") != std::string::npos);
        if (rescale) {
            m_inputs_qqbb["RWpTM_qqbb"] = m_inputs_qqbb["RWpTM_qqbb"]*1000;
            m_inputs_qqbb["RHpTM_qqbb"] = m_inputs_qqbb["RHpTM_qqbb"]*1000;
        }
        if ((WriteOnlyCorrectCategory) && !((selection_category == 3) || (selection_category == 9))) {
            qqbbNNs[kv_pair.first] = -1;
        }
        else{
            if (kv_pair.first.find("pNN800") != std::string::npos) {
                m_inputs_qqbb["mass"] = 0.8;
                mm_inputs["inputs_node"] = m_inputs_qqbb;
                outputs = kv_pair.second->compute(mm_inputs);
            }
            else if (kv_pair.first.find("pNN900") != std::string::npos) {
                m_inputs_qqbb["mass"] = 0.9;
                mm_inputs["inputs_node"] = m_inputs_qqbb;
                outputs = kv_pair.second->compute(mm_inputs);
            }
            else if (kv_pair.first.find("pNN1000") != std::string::npos) {
                m_inputs_qqbb["mass"] = 1.0;
                mm_inputs["inputs_node"] = m_inputs_qqbb;
                outputs = kv_pair.second->compute(mm_inputs);
            }
            else if (kv_pair.first.find("pNN1200") != std::string::npos) {
                m_inputs_qqbb["mass"] = 1.2;
                mm_inputs["inputs_node"] = m_inputs_qqbb;
                outputs = kv_pair.second->compute(mm_inputs);
            }
            else if (kv_pair.first.find("pNN1400") != std::string::npos) {
                m_inputs_qqbb["mass"] = 1.4;
                mm_inputs["inputs_node"] = m_inputs_qqbb;
                outputs = kv_pair.second->compute(mm_inputs);
            }
            else if (kv_pair.first.find("pNN1600") != std::string::npos) {
                m_inputs_qqbb["mass"] = 1.6;
                mm_inputs["inputs_node"] = m_inputs_qqbb;
                outputs = kv_pair.second->compute(mm_inputs);
            }
            else if (kv_pair.first.find("pNN1800") != std::string::npos) {
                m_inputs_qqbb["mass"] = 1.8;
                mm_inputs["inputs_node"] = m_inputs_qqbb;
                outputs = kv_pair.second->compute(mm_inputs);
            }
            else if (kv_pair.first.find("pNN2000") != std::string::npos) {
                m_inputs_qqbb["mass"] = 2.0;
                mm_inputs["inputs_node"] = m_inputs_qqbb;
                outputs = kv_pair.second->compute(mm_inputs);
            }
            else if (kv_pair.first.find("pNN2500") != std::string::npos) {
                m_inputs_qqbb["mass"] = 2.5;
                mm_inputs["inputs_node"] = m_inputs_qqbb;
                outputs = kv_pair.second->compute(mm_inputs);
            }
            else if (kv_pair.first.find("pNN3000") != std::string::npos) {
                m_inputs_qqbb["mass"] = 3.0;
                mm_inputs["inputs_node"] = m_inputs_qqbb;
                outputs = kv_pair.second->compute(mm_inputs);
            }
            else{
                mm_inputs["inputs_node"] = m_inputs_qqbb;
                if (debugMode) {
                    std::cout << "\t" << "Inputs are: " << std::endl;
                    for (auto &ipt: mm_inputs["inputs_node"]){
                        std::cout << "\t\t" << ipt.first << ": " << ipt.second << std::endl;
                    }
                }
                outputs = kv_pair.second->compute(mm_inputs);
                // if (debugMode) std::cout << "\t" << "Value found is: " << outputs[0].second << std::endl;
            }
            for (auto &opt: outputs){
                qqbbNNs[kv_pair.first] = opt.second;
                if (debugMode) std::cout << "\t" << "Value found is: " << opt.second << std::endl;
            }
            if (doCombined){
                basename_delim_pos = kv_pair.first.find_last_of("_");
                NN_basename = kv_pair.first.substr(0, basename_delim_pos);
                if ((eventNumber % 2 == 0) && (kv_pair.first.find("OddTrained") != std::string::npos)){
                    NN_foldname = NN_basename + "_OddTrained";
                    if (debugMode) std::cout << "\t" << "Event number: " << eventNumber << ", so saving " << NN_foldname << std::endl;
                    qqbbNNs[NN_basename + "_Combined"] = qqbbNNs[kv_pair.first];
                }
                else if ((eventNumber % 2 == 1) && (kv_pair.first.find("EvenTrained") != std::string::npos)){
                    NN_foldname = NN_basename + "_EvenTrained";
                    if (debugMode) std::cout << "\t" << "Event number: " << eventNumber << ", so saving " << NN_foldname << std::endl;
                    qqbbNNs[NN_basename + "_Combined"] = qqbbNNs[kv_pair.first];
                }
                else{
                    if (debugMode) std::cout << "\t" << "Event number: " << eventNumber << ", so NOT saving " << kv_pair.first << std::endl;
                }
            }
        }
        // Fix a couple of values in terms of GeV vs. MeV
        if (rescale) {
            m_inputs_qqbb["RWpTM_qqbb"] = m_inputs_qqbb["RWpTM_qqbb"]/1000;
            m_inputs_qqbb["RHpTM_qqbb"] = m_inputs_qqbb["RHpTM_qqbb"]/1000;
        }
    }
    if (doCombined){
        // First lvbb
        if ((WriteOnlyCorrectCategory) && !((selection_category == 0) || (selection_category == 8) || (selection_category == 10))) {
            lvbbNNs["NN_lvbb_Final_Combined"] = -1;
        }
        else{
            if (eventNumber % 2 == 0){
                lvbbNNs["NN_lvbb_Final_Combined"] = lvbbNNs["NN_lvbb_Final_OddTrained"];
            }
            else{
                lvbbNNs["NN_lvbb_Final_Combined"] = lvbbNNs["NN_lvbb_Final_EvenTrained"];
            }
        }
        // Now qqbb
        if ((WriteOnlyCorrectCategory) && !((selection_category == 3) || (selection_category == 9))) {
            qqbbNNs["NN_qqbb_Final_Combined"] = -1;
        }
        else{
            if (eventNumber % 2 == 0){
                qqbbNNs["NN_qqbb_Final_Combined"] = qqbbNNs["NN_qqbb_Final_OddTrained"];
            }
            else{
                qqbbNNs["NN_qqbb_Final_Combined"] = qqbbNNs["NN_qqbb_Final_EvenTrained"];
            }
        }
    }
    if ((selection_category == 0) || (selection_category == 8) || (selection_category == 10)){
        combined_category = 0;
    }
    else if ((selection_category == 3) || (selection_category == 9)){
        combined_category = 3;
    }
    else{
        combined_category = -2;
    }

}

bool EventLoop::LowLevel_MatchTruthParticles(){
    // Match the low-level reco objects (elec, muon, neutrino, jets) to whether they appear to come from the H+ -> Wh in the truth level...
    if (debugMode) std::cout << "\t" << "Entering LowLevel_MatchTruthParticles" << std::endl;
    // First find the reco large-R jet or small-R jets which most closesly match the truth Higgs
    int idx_counter = 0;
    float delta_R_TruthHiggs = -1;
    int best_H_ljet_idx = -1, best_H_sjet_idx1 = -1, best_H_sjet_idx2 = -1;
    bool best_bb_is_largeR = false;
    if (false) // Old (2025/02/27) method of matching Higgs large-R jet/small-R jets
    {
        for (const auto &particle : particles) {
            if ((particle.type == 5)){
                if (((ll_truth_Higgs.DeltaR(particle.p4)) < delta_R_TruthHiggs) || (delta_R_TruthHiggs==-1)){
                    delta_R_TruthHiggs = ll_truth_Higgs.DeltaR(particle.p4);
                    best_H_ljet_idx = idx_counter;
                    best_bb_is_largeR = true;
                }
            }
            idx_counter++;
        }
        if (best_H_ljet_idx == -1){ // No Xbb-tagged jets, so loop through non-Xbb tagged jets and small-jet pairs for H. Shouldn't really use this case but I'll do it anyway since we might want to use later
            idx_counter = 0;
            for (const auto &particle : particles) {
                if ((particle.type == 3)){
                    if (((ll_truth_Higgs.DeltaR(particle.p4)) < delta_R_TruthHiggs) || (delta_R_TruthHiggs==-1)){
                        delta_R_TruthHiggs = ll_truth_Higgs.DeltaR(particle.p4);
                        best_H_ljet_idx = idx_counter;
                        best_bb_is_largeR = true;
                    }
                }
                idx_counter++;
            }
            for (int i1=0; i1 < particles.size(); i1++){
                if (particles[i1].type == 4){
                    for(int i2=i1+1; i2 < particles.size(); i2++){
                        if (particles[i2].type == 4){
                                if (((ll_truth_Higgs.DeltaR(particles[i1].p4+particles[i2].p4)) < delta_R_TruthHiggs) || (delta_R_TruthHiggs==-1)){
                                delta_R_TruthHiggs = ll_truth_Higgs.DeltaR(particles[i1].p4+particles[i2].p4);
                                best_H_sjet_idx1 = i1;
                                best_H_sjet_idx2 = i2;
                                best_bb_is_largeR = false;
                            }
                        }
                    }
                }
            }
        }
        if (delta_R_TruthHiggs != -1){
            if (best_bb_is_largeR){
                particles[best_H_ljet_idx].trueInclusion = 1;
            } else{
                particles[best_H_sjet_idx1].trueInclusion = 1;
                particles[best_H_sjet_idx2].trueInclusion = 1;
            }
        } else{
            return false;
        }
    } else // New (2025/02/28) method of matching Higgs large-R jet/small-R jets
    {
        for (const auto &particle : particles) {
            if ((particle.type == 5)){
                if (((ll_truth_Higgs.DeltaR(particle.p4)) < delta_R_TruthHiggs) || (delta_R_TruthHiggs==-1)){
                    delta_R_TruthHiggs = ll_truth_Higgs.DeltaR(particle.p4);
                    best_H_ljet_idx = idx_counter;
                    best_bb_is_largeR = true;
                }
            }
            idx_counter++;
        }
        idx_counter = 0;
        for (const auto &particle : particles) {
            if ((particle.type == 3)){
                if (((ll_truth_Higgs.DeltaR(particle.p4)) < delta_R_TruthHiggs) || (delta_R_TruthHiggs==-1)){
                    delta_R_TruthHiggs = ll_truth_Higgs.DeltaR(particle.p4);
                    best_H_ljet_idx = idx_counter;
                    best_bb_is_largeR = true;
                }
            }
            idx_counter++;
        }
        for (int i1=0; i1 < particles.size(); i1++){
            if (particles.at(i1).type == 4){
                for(int i2=i1+1; i2 < particles.size(); i2++){
                    if (particles.at(i2).type == 4){
                            if (((ll_truth_Higgs.DeltaR(particles.at(i1).p4+particles.at(i2).p4)) < delta_R_TruthHiggs) || (delta_R_TruthHiggs==-1)){
                            delta_R_TruthHiggs = ll_truth_Higgs.DeltaR(particles.at(i1).p4+particles.at(i2).p4);
                            best_H_sjet_idx1 = i1;
                            best_H_sjet_idx2 = i2;
                            best_bb_is_largeR = false;
                        }
                    }
                }
            }
        }
        if (delta_R_TruthHiggs != -1){
            if (best_bb_is_largeR){
                if (debugMode) std::cout << "\t\t" << "Truth Higgs being matched to ljet" << best_H_ljet_idx << std::endl;
                particles.at(best_H_ljet_idx).trueInclusion = 1;
            } else{
                if (debugMode) std::cout << "\t\t" << "Truth Higgs being matched to sjets" << best_H_sjet_idx1 << " and " << best_H_sjet_idx1 << std::endl;
                particles.at(best_H_sjet_idx1).trueInclusion = 1;
                particles.at(best_H_sjet_idx2).trueInclusion = 1;
            }
        } else{
            if (debugMode) std::cout << "\t\t" << "Leaving LowLevel_MatchTruthParticles with Early (H matching) failure" << std::endl;
            return false;
        }
    }

    // Now the find the reco large-R jet or small-R jets or lep+neutrino combo which most closesly match the truth W, depending on truth decay mode
    if (truth_decay_mode==1){
        // Simply set the lepton and neutrino to be on
        particles.at(0).trueInclusion = 3;
        particles.at(1).trueInclusion = 3;
    } else if (truth_decay_mode==2){
        // w -> qq
        // First look for large-R jet
        idx_counter = 0;
        bool best_Wqq_is_largeR = false;
        float delta_R_TruthW = -1;
        int best_W_ljet_idx = -1, best_W_sjet_idx1 = -1, best_W_sjet_idx2 = -1;
        for (const auto &particle : particles) {
            if  ((idx_counter!=best_H_ljet_idx) && ((particle.type == 3) || (particle.type == 5))){
                if (((ll_truth_W.DeltaR(particle.p4)) < delta_R_TruthW) || (delta_R_TruthW==-1)){
                    delta_R_TruthW = ll_truth_W.DeltaR(particle.p4);
                    best_W_ljet_idx = idx_counter;
                    best_Wqq_is_largeR = true;
                }
            }
            idx_counter++;
        }
        // Now try with W as pairs of small-R jets
        for (int i1=0; i1 < particles.size(); i1++){
            if (particles.at(i1).type == 4){
                for(int i2=i1+1; i2 < particles.size(); i2++){
                    if (particles.at(i2).type == 4){
                        if ((i1!=best_H_sjet_idx1) || (i2!=best_H_sjet_idx2)){ // Check we're not overlapping with small-R jet h->bb if that was selected
                            // if ((abs((particles[i1].p4+particles[i2].p4).M()-80.36e3) < best_W_diff) || (best_W_diff==-1)){
                            if (((ll_truth_W.DeltaR(particles.at(i1).p4+particles.at(i2).p4)) < delta_R_TruthW) || (delta_R_TruthW==-1)){
                                delta_R_TruthW = ll_truth_W.DeltaR(particles.at(i1).p4+particles.at(i2).p4);
                                best_W_sjet_idx1 = i1;
                                best_W_sjet_idx2 = i2;
                                best_Wqq_is_largeR = false;
                            }
                        }
                    }
                }
            }
        }
        if (delta_R_TruthW != -1){
            if (best_Wqq_is_largeR){
                if (debugMode) std::cout << "\t\t" << "Truth W being matched to ljet" << best_W_ljet_idx << std::endl;
                // particles[best_W_ljet_idx].trueInclusion = 2;
                particles.at(best_W_ljet_idx).trueInclusion = 2;
            } else{
                if (debugMode) std::cout << "\t\t" << "Truth W being matched to sjets" << best_W_sjet_idx1 << " and " << best_W_sjet_idx2 << std::endl;
                // particles[best_W_sjet_idx1].trueInclusion = 2;
                // particles[best_W_sjet_idx2].trueInclusion = 2;
                particles.at(best_W_sjet_idx1).trueInclusion = 2;
                particles.at(best_W_sjet_idx2).trueInclusion = 2;
            }
        } else{
            if (debugMode) std::cout << "\t\t" << "Leaving LowLevel_MatchTruthParticles with Early (W matching) failure" << std::endl;
            return false;
        }
    } else{ // Should not be able to get here
        assert((false));
    }
    if (debugMode) std::cout << "\t" << "Leaving LowLevel_MatchTruthParticles" << std::endl;
    return true;
}

std::tuple<float, float, float> EventLoop::LowLevel_GetBestWhMasses(){
    if (debugMode) std::cout << "\t" << "Entering LowLevel_GetBestWhMasses" << std::endl;
    TLorentzVector best_H, best_W_lv, best_W_qq;
    float best_qqbb=-1, best_lvbb=-1, best_bb = -1, best_H_diff, best_W_diff;
    int best_H_ljet_idx = -1, best_H_sjet_idx1 = -1, best_H_sjet_idx2 = -1, best_W_ljet_idx = -1, best_W_sjet_idx1 = -1, best_W_sjet_idx2 = -1;
    best_H_diff = -1; // initialize to -1 so we know it's an error if it is -1 since should be abs(diff)
    best_W_diff = -1; // initialize to -1 so we know it's an error if it is -1 since should be abs(diff)
    // for (const auto &particle : particles) {
    int idx_counter = 0;
    bool best_bb_is_largeR = false;
    for (const auto &particle : particles) {
        if ((particle.type == 5)){
            if ((abs(particle.p4.M()-125e3) < best_H_diff) || (best_H_diff==-1)){
                best_H_diff = abs(particle.p4.M()-125e3);
                best_H.SetPtEtaPhiE(particle.p4.Pt(), particle.p4.Eta(), particle.p4.Phi(), particle.p4.E());
                best_H_ljet_idx = idx_counter;
                best_bb = best_H.M();
                best_bb_is_largeR = true;
            }
        }
        idx_counter++;
    }
    if (best_H_ljet_idx == -1){ // No Xbb-tagged jets, so loop through non-Xbb tagged jets and small-jet pairs for H. Shouldn't really use this case but I'll do it anyway since we might want to use later
        int idx_counter = 0;
        for (const auto &particle : particles) {
            if ((particle.type == 3)){
                if ((abs(particle.p4.M()-125e3) < best_H_diff) || (best_H_diff==-1)){
                    best_H_diff = abs(particle.p4.M()-125e3);
                    best_H.SetPtEtaPhiE(particle.p4.Pt(), particle.p4.Eta(), particle.p4.Phi(), particle.p4.E());
                    best_H_ljet_idx = idx_counter;
                    best_bb = best_H.M();
                    best_bb_is_largeR = true;
                }
            }
            idx_counter++;
        }
        for (int i1=0; i1 < particles.size(); i1++){
            if (particles[i1].type == 4){
                for(int i2=i1+1; i2 < particles.size(); i2++){
                    if (particles[i2].type == 4){
                        if ((abs((particles[i1].p4+particles[i2].p4).M()-125e3) < best_H_diff) || (best_H_diff==-1)){
                            best_H_diff = abs((particles[i1].p4+particles[i2].p4).M()-125e3);
                            best_H = particles[i1].p4 + particles[i2].p4;
                            best_H_sjet_idx1 = i1;
                            best_H_sjet_idx2 = i2;
                            best_bb = best_H.M();
                            best_bb_is_largeR = false;
                        }
                    }
                }
            }
        }
    }
    if (best_H_ljet_idx!=-1){
        if (best_bb_is_largeR){
            particles.at(best_H_ljet_idx).recoInclusion = 1;
        } else{
            particles.at(best_H_sjet_idx1).recoInclusion = 1;
            particles.at(best_H_sjet_idx2).recoInclusion = 1;
        }
    }

    // Now find best W indices as a single large-R jet (separate from Higgs) if one exists
    idx_counter = 0;
    bool best_Wqq_is_largeR = false;
    for (const auto &particle : particles) {
        if  ((idx_counter!=best_H_ljet_idx) && ((particle.type == 3) || (particle.type == 5))){
            // if ((abs(particle.p4.M()-80.36e3) < best_W_diff) || (best_W_diff==-1)){
            if (((particle.p4.M() < 110e3) && (particle.p4.Pt() > best_W_diff)) || (best_W_diff==-1)){
                // best_W_diff = abs(particle.p4.M()-80.36e3);
                best_W_diff = particle.p4.Pt();
                best_W_qq.SetPtEtaPhiE(particle.p4.Pt(), particle.p4.Eta(), particle.p4.Phi(), particle.p4.E());
                best_W_ljet_idx = idx_counter;
                best_Wqq_is_largeR = true;
            }
        }
        idx_counter++;
    }
    // Now try with W as pairs of small-R jets
    for (int i1=0; i1 < particles.size(); i1++){
        if (particles[i1].type == 4){
            for(int i2=i1+1; i2 < particles.size(); i2++){
                if (particles[i2].type == 4){
                    if ((i1!=best_H_sjet_idx1) || (i2!=best_H_sjet_idx2)){ // Check we're not overlapping with small-R jet h->bb if that was selected
                        // if ((abs((particles[i1].p4+particles[i2].p4).M()-80.36e3) < best_W_diff) || (best_W_diff==-1)){
                        if ((((particles[i1].p4+particles[i2].p4).M() < 110e3) && ((particles[i1].p4+particles[i2].p4).Pt() > best_W_diff)) || (best_W_diff==-1)){
                            best_W_diff = (particles[i1].p4+particles[i2].p4).Pt();
                            best_W_qq = particles[i1].p4 + particles[i2].p4;
                            best_W_sjet_idx1 = i1;
                            best_W_sjet_idx2 = i2;
                            best_Wqq_is_largeR = false;
                        }
                    }
                }
            }
        }
    }
    if (best_W_diff!=-1){
        if (best_Wqq_is_largeR){
            particles.at(best_W_ljet_idx).recoInclusion = 2;
        } else{
            particles.at(best_W_sjet_idx1).recoInclusion = 2;
            particles.at(best_W_sjet_idx2).recoInclusion = 2;
        }
        best_qqbb = (best_H + best_W_qq).M();
        best_mWqq = best_W_qq.M();
    }

    TLorentzVector lep, neut;
    idx_counter = 0;
    for (const auto &particle : particles) {
        if ((particle.type == 0) || (particle.type == 1)){
            particles[idx_counter].recoInclusion = 3;
            lep = particle.p4;
        }
        if (particle.type==2){
            particles[idx_counter].recoInclusion = 3;
            neut = particle.p4;
        }
        idx_counter+=1;
    }
    best_lvbb = (best_H + neut + lep).M();
    best_mWlv = (neut + lep).M();
    if (false){ // Testing 15/01/25
        std::cout << "RECO W_lv : (" << (neut + lep).Pt() << ", " << (neut + lep).Eta() << ", " << (neut + lep).Phi() << ", " << (neut + lep).M() << std::endl;
        std::cout << "RECO W_qq : (" << best_W_qq.Pt() << ", " << best_W_qq.Eta() << ", " << best_W_qq.Phi() << ", " << best_W_qq.M() << std::endl;
        std::cout << "RECO BestWh_qqbb : " << best_qqbb/1e3 << std::endl;
        if ((abs(best_qqbb-1600e3)>500e3) && (truth_decay_mode==2)){
            std::cout << "HERE!" << std::endl;
        }
        std::cout << "RECO BestWh_lvbb : " << best_lvbb/1e3 << std::endl;
        std::cout << "RECO Best_h_bb : " << best_bb/1e3 << std::endl;
    }
    if (debugMode) std::cout << "\t" << "Leaving LowLevel_GetBestWhMasses" << std::endl;
    return std::make_tuple(best_bb, best_qqbb, best_lvbb);
}

int EventLoop::LowLevel_CountLeptons(){
    if (debugMode) std::cout << "\t" << "Entering LowLevel_CountLeptons" << std::endl;
    // Check if truth information exists
    if (truth_pt->empty() || truth_eta->empty() || truth_phi->empty() || 
        truth_m->empty() || truth_pdgid->empty()) {
        return -1;
    }
    const double MIN_DELTAR = 0.01; // Minimum deltaR to consider particles distinct
    // Create vectors to store particle information
    std::vector<TLorentzVector> particleP4;
    std::vector<int> leptonPdgIds;
    for (size_t i = 0; i < truth_pdgid->size(); i++) {
        if (truth_pdgid->at(i) == -15){
            return -2;
        }
    }
    // Fill lepton vectors with truth information
    for (size_t i = 0; i < truth_pt->size(); i++) {
        if (abs(truth_pdgid->at(i)) == 11 || abs(truth_pdgid->at(i)) == 13)
        {
            TLorentzVector p4;
            p4.SetPtEtaPhiM(truth_pt->at(i), truth_eta->at(i), 
                            truth_phi->at(i), truth_m->at(i));
            particleP4.push_back(p4);
            leptonPdgIds.push_back(truth_pdgid->at(i));
        }
    }
    // Mark unique leptons
    std::vector<bool> is_unique(particleP4.size(), true);
    for(size_t i = 0; i < particleP4.size(); i++) {
        if(!is_unique[i]) continue; // already marked as duplicate
        for(size_t j = i + 1; j < particleP4.size(); j++) {
            if(!is_unique[j]) continue; // already marked as duplicate
            // Check if same type (same pdgid) and close in deltaR
            if(leptonPdgIds[i] == leptonPdgIds[j] && 
               particleP4[i].DeltaR(particleP4[j]) < MIN_DELTAR) {
                is_unique[j] = false; // mark as duplicate
            }
        }
    }
    // Count unique leptons
    int leptonCount = std::count(is_unique.begin(), is_unique.end(), true);
    if (debugMode) std::cout << "\t" << "Leaving LowLevel_CountLeptons" << std::endl;
    return leptonCount;
}

int EventLoop::LowLevel_ClassifyDecayType(){
    if (debugMode) std::cout << "\t" << "Entering LowLevel_ClassifyDecayType" << std::endl;
    const double W_MASS = 80.379e3; // GeV to MeV
    const double W_MASS_WINDOW = 20.0e3; // GeV to MeV
    // Check if truth information exists
    if (truth_pt->empty() || truth_eta->empty() || truth_phi->empty() || 
        truth_m->empty() || truth_pdgid->empty()) {
        return -3;
    }

    // Create vectors to store particle information
    std::vector<TLorentzVector> particleP4;
    std::vector<int> pdgIds;
    // Fill vectors with truth information
    for (size_t i = 0; i < truth_pt->size(); i++) {
        TLorentzVector p4;
        p4.SetPtEtaPhiM(truth_pt->at(i), truth_eta->at(i), 
                        truth_phi->at(i), truth_m->at(i));
        particleP4.push_back(p4);
        pdgIds.push_back(truth_pdgid->at(i));
    }

    if (false){ // Testing 15/01/25
        for (size_t i = 0; i < pdgIds.size(); i++) {
            if (pdgIds[i] == 24) {
                std::cout << "TRUTH W+ : (" << particleP4[i].Pt() << ", " << particleP4[i].Eta() << ", " << particleP4[i].Phi() << ", " << particleP4[i].M() << std::endl;
            }
            if (pdgIds[i] == -24) {
                std::cout << "TRUTH W- : (" << particleP4[i].Pt() << ", " << particleP4[i].Eta() << ", " << particleP4[i].Phi() << ", " << particleP4[i].M() << std::endl;
            }
        }
    }

    // Find charged Higgs (pdgId = ±37)
    int chHiggsIdx = -1;
    TLorentzVector chHiggsP4;
    for (size_t i = 0; i < pdgIds.size(); i++) {
        if (abs(pdgIds[i]) == 37) {
            chHiggsIdx = i;
            chHiggsP4 = particleP4[i];
            break;
        }
    }
    if (chHiggsIdx == -1) return -4;
    // Find SM Higgs (pdgId = 25)
    int smHiggsIdx = -1;
    TLorentzVector smHiggsP4;
    for (size_t i = 0; i < pdgIds.size(); i++) {
        if (pdgIds[i] == 25) {
            smHiggsIdx = i;
            smHiggsP4 = particleP4[i];
            break;
        }
    }
    if (smHiggsIdx == -1) return -5;
    ll_truth_Higgs.SetPtEtaPhiM(smHiggsP4.Pt(), smHiggsP4.Eta(), smHiggsP4.Phi(), smHiggsP4.M());
    // Check for b-quark pair from Higgs decay
    bool foundBPair = false;
    int bCount = 0;
    for (int pdgId : pdgIds) {
        if (abs(pdgId) == 5) bCount++;
    }
    if (bCount >= 2) foundBPair = true;
    if (!foundBPair) return -6;
    // truth_Higgs_reco = ???;
    // Find W boson that best reconstructs charged Higgs when combined with SM Higgs
    double bestDeltaM = 1e12;
    int bestWIdx = -1;
    TLorentzVector bestWP4;
    for (size_t i = 0; i < pdgIds.size(); i++) {
        // if (abs(pdgIds[i]) == 24) {
        if (pdgIds[i] == 24) {
            double deltaM = abs((particleP4[i] + smHiggsP4).M() - chHiggsP4.M());
            if (deltaM < bestDeltaM) {
                bestDeltaM = deltaM;
                bestWIdx = i;
                bestWP4 = particleP4[i];
            }
        }
    }
    if (bestWIdx == -1) return -7;
    ll_truth_W.SetPtEtaPhiM(bestWP4.Pt(), bestWP4.Eta(), bestWP4.Phi(), bestWP4.M());
    int decay_type = 0;
    int lepton_idx = -1;
    if (false){ // Old version, doesn't account for 2 lepton case
        // First find the lepton
        for(size_t i = 0; i < truth_pdgid->size(); i++) {
            int pdgid = (*truth_pdgid)[i];
            // Check for electron (11) or muon (13)
            if(abs(pdgid) == 11 || abs(pdgid) == 13) {
                lepton_idx = i;
                // Set decay type based on charge (negative pdgid = positive particle)
                decay_type = (pdgid > 0) ? 2 : 1;
                break;
            }
        }
    } else if (false) { // Newer but still old version, checks for which truth lepton most closely matches the event-level lepton
        // First find the lepton
        float min_delta_R = -1;
        for(size_t i = 0; i < truth_pdgid->size(); i++) {
            int pdgid = (*truth_pdgid)[i];
            // Check for electron (11) or muon (13)
            if(abs(pdgid) == 11 || abs(pdgid) == 13) {
                if ((min_delta_R==-1) || (particles.at(0).p4.DeltaR(particleP4.at(i))) < min_delta_R){
                    lepton_idx = i;
                    min_delta_R = (particles.at(0).p4.DeltaR(particleP4.at(i)));
                    // Set decay type based on charge (negative pdgid = positive particle)
                    decay_type = (pdgid > 0) ? 2 : 1;
                }
            }
        }
    } else { // New version, checks the W boson decay by trying to create W from neutrino+lep OR quarks
        float min_delta_R = -1;
        // First try with lepton + neutrino pairs
        if (debugMode) std::cout << "\t" << "Looping over leptons" << std::endl;
        for(size_t i = 0; i < truth_pdgid->size(); i++) {
            int pdgid = (*truth_pdgid)[i];
            // Check for electron (11) or muon (13)
            if(pdgid == -11 || pdgid == -13) {
                if (debugMode) std::cout << "\t" << "Found lepton with pdgid=" << pdgid << std::endl;
                for(size_t j = 0; j < truth_pdgid->size(); j++) {
                    int pdgid2 = (*truth_pdgid)[j];
                    if(((pdgid==-11)&&(pdgid2==12)) || ((pdgid==-13)&&(pdgid2==14))) {
                        if (debugMode) std::cout << "\t" << "Found neutrino with pdgid=" << pdgid << std::endl;
                        if ((ll_truth_W.DeltaR(particleP4.at(i) + particleP4.at(j)) < min_delta_R) || (min_delta_R==-1)){
                            lepton_idx = i;
                            min_delta_R = ll_truth_W.DeltaR(particleP4.at(i) + particleP4.at(j));
                            decay_type = 1;
                        }
                    }
                }
            }
        }
        // Now try with quarks
        int best_j1_idx = -1;
        int best_j2_idx = -1;
        for(size_t i = 0; i < truth_pdgid->size(); i++) {
            // Skip if not a quark (assuming quarks have pdgid < 6)
            if(abs((*truth_pdgid)[i]) >= 6) continue;
            for(size_t j = i + 1; j < truth_pdgid->size(); j++) {
                if(abs((*truth_pdgid)[j]) >= 6) continue;
                // TLorentzVector dijet;
                double mass_diff = abs((particleP4.at(i) + particleP4.at(j)).M() - W_MASS);
                if(mass_diff < W_MASS_WINDOW) {
                    double deltaR = ll_truth_W.DeltaR(particleP4.at(i) + particleP4.at(j));
                    if((deltaR < min_delta_R) || (min_delta_R==-1)) {
                        min_delta_R = deltaR;
                        best_j1_idx = i;
                        best_j2_idx = j;
                        decay_type = 2;
                    }
                }
            }
        }
    }

    // if(lepton_idx == -1) return -8; // No lepton found
    // // Create lepton TLorentzVector
    // TLorentzVector lepton;
    // lepton.SetPtEtaPhiM((*truth_pt)[lepton_idx],
    //                     (*truth_eta)[lepton_idx],
    //                     (*truth_phi)[lepton_idx],
    //                     (*truth_m)[lepton_idx]);
    // if(decay_type == 1) {
    //     // Look for corresponding neutrino
    //     int target_nu_pdgid;
    //     if(abs((*truth_pdgid)[lepton_idx]) == 11) {
    //         target_nu_pdgid = 12; // electron neutrino
    //     } else {
    //         target_nu_pdgid = 14; // muon neutrino
    //     }
        
    //     int nu_idx = -1;
    //     for(size_t i = 0; i < truth_pdgid->size(); i++) {
    //         if((*truth_pdgid)[i] == target_nu_pdgid) {
    //             nu_idx = i;
    //             break;
    //         }
    //     }
    //     if(nu_idx == -1) return -1; // No matching neutrino found
    //     // Create neutrino TLorentzVector and sum with lepton
    //     TLorentzVector neutrino;
    //     neutrino.SetPtEtaPhiM((*truth_pt)[nu_idx],
    //                          (*truth_eta)[nu_idx],
    //                          (*truth_phi)[nu_idx],
    //                          (*truth_m)[nu_idx]);
    //     ll_truth_W_reco = lepton + neutrino;
    // } else if(decay_type == 2) {
    //     // Look for jet pair with mass close to W mass
    //     double best_deltaR = 999.0;
    //     int best_j1_idx = -1;
    //     int best_j2_idx = -1;
    //     for(size_t i = 0; i < truth_pdgid->size(); i++) {
    //         // Skip if not a jet (assuming jets have pdgid < 7)
    //         if(abs((*truth_pdgid)[i]) >= 6) continue;
    //         TLorentzVector jet1;
    //         jet1.SetPtEtaPhiM((*truth_pt)[i],
    //                          (*truth_eta)[i],
    //                          (*truth_phi)[i],
    //                          (*truth_m)[i]);
    //         for(size_t j = i + 1; j < truth_pdgid->size(); j++) {
    //             if(abs((*truth_pdgid)[j]) >= 6) continue;
    //             TLorentzVector jet2;
    //             jet2.SetPtEtaPhiM((*truth_pt)[j],
    //                              (*truth_eta)[j],
    //                              (*truth_phi)[j],
    //                              (*truth_m)[j]);
    //             TLorentzVector dijet = jet1 + jet2;
    //             double mass_diff = abs(dijet.M() - W_MASS);
    //             if(mass_diff < W_MASS_WINDOW) {
    //                 double deltaR = dijet.DeltaR(ll_truth_W);
    //                 if(deltaR < best_deltaR) {
    //                     best_deltaR = deltaR;
    //                     best_j1_idx = i;
    //                     best_j2_idx = j;
    //                 }
    //             }
    //         }
    //     }
    //     if(best_j1_idx == -1 || best_j2_idx == -1) return -2; // No suitable jet pair found
    //     // Reconstruct W from best jet pair
    //     TLorentzVector jet1, jet2;
    //     jet1.SetPtEtaPhiM((*truth_pt)[best_j1_idx],
    //                       (*truth_eta)[best_j1_idx],
    //                       (*truth_phi)[best_j1_idx],
    //                       (*truth_m)[best_j1_idx]);
    //     jet2.SetPtEtaPhiM((*truth_pt)[best_j2_idx],
    //                       (*truth_eta)[best_j2_idx],
    //                       (*truth_phi)[best_j2_idx],
    //                       (*truth_m)[best_j2_idx]);
        
    //     ll_truth_W_reco = jet1 + jet2;
    // }
    for(size_t i = 0; i < truth_pdgid->size(); i++) {
        if (((*truth_pdgid)[i] == -15) and (decay_type == 1)) return -9; // Tau+ in the event truth-particles so this case is tricky, discount it
        if (((*truth_pdgid)[i] == -15) and (decay_type == 2)) return -10; // Tau+ in the event truth-particles so this case is tricky, discount it
    }
    if (debugMode) std::cout << "\t" << "Leaving LowLevel_ClassifyDecayType" << std::endl;
    return decay_type;
}


bool EventLoop::LowLevel_Loop(){
    if (debugMode) std::cout << "\t" << "Entering LowLevel_Loop" << std::endl;
    // Clear vectors for new event
    // ResetVariables();
    ll_particle_px.clear();
    ll_particle_py.clear();
    ll_particle_pz.clear();
    ll_particle_e.clear();
    ll_particle_type.clear();
    ll_particle_tagInfo.clear();
    ll_particle_recoInclusion.clear();
    ll_particle_trueInclusion.clear();
    ll_truth_Higgs.SetPtEtaPhiE(0,0,0,0);
    ll_truth_Higgs_old.SetPtEtaPhiE(0,0,0,0);
    ll_truth_W.SetPtEtaPhiE(0,0,0,0);
    ll_truth_W_old.SetPtEtaPhiE(0,0,0,0);
    ll_truth_W_reco.SetPtEtaPhiE(0,0,0,0);
    ll_truth_W_reco_old.SetPtEtaPhiE(0,0,0,0);
    particles.clear();
    // Get the lepton
    Particle lepton;
    if (!el_pt->empty()) {
        lepton.p4.SetPtEtaPhiE(el_pt->at(0), el_eta->at(0), el_phi->at(0), el_e->at(0));
        lepton.type = 0; // electron
    } else if (!mu_pt->empty()) {
        lepton.p4.SetPtEtaPhiE(mu_pt->at(0), mu_eta->at(0), mu_phi->at(0), mu_e->at(0));
        lepton.type = 1; // muon
    }
    lepton.tagInfo = 0;
    lepton.recoInclusion = 0;
    lepton.trueInclusion = 0;
    particles.push_back(lepton);
    // Process neutrino - now using lepton information TODO Maybe use the same neutrino calculation as the orignal method here?
    // Particle neutrino = LowLevel_createNeutrino(met_met, met_phi, lepton.p4);
    MET.SetPtEtaPhiM(met_met, 0., met_phi, 0.); // TODO check this is correct/reasonable
    std::vector<TLorentzVector *> neutrinoVector = GetNeutrinos(&lepton.p4, &MET);
    Particle neutrino;
    neutrino.type = 2;
    neutrino.p4.SetPtEtaPhiE(neutrinoVector.at(0)->Pt(), neutrinoVector.at(0)->Eta(), neutrinoVector.at(0)->Phi(), neutrinoVector.at(0)->E());
    neutrino.tagInfo = 0;
    neutrino.recoInclusion = 0;
    neutrino.trueInclusion = 0;
    particles.push_back(neutrino);
    // Process large-radius jets
    std::vector<TLorentzVector> ljetCandidates;
    float DxbbThreshold = 2.44;
    float DXbb;
    for (size_t j = 0; j < ljet_pt->size(); j++) {
        Particle ljet;
        ljet.p4.SetPtEtaPhiE(ljet_pt->at(j), ljet_eta->at(j), ljet_phi->at(j), ljet_e->at(j));
        if ((LowLevelDeltaRLepLjetCut) && (fabs(ljet.p4.DeltaR(lepton.p4)) < 1.0)){
            continue;
        }
        if ((LowLevelLjetPtCut) && (ljet.p4.Pt() < 250e3)){
            continue;
        }
        if ((LowLevelLjetMassCut) && ((ljet.p4.M() < 50e3) || (ljet.p4.M() > 250e3))){
            continue;
        }
        DXbb = log(ljet_Xbb2020v3_Higgs->at(j)/(0.25*ljet_Xbb2020v3_Top->at(j) + 0.75*ljet_Xbb2020v3_QCD->at(j)));
        if (DXbb >= DxbbThreshold){
            ljet.type = 5; // Xbb-tagged large-R jet
        } else{
            ljet.type = 3; // Non-Xbb-tagged large-R jet
        }
        ljet.tagInfo = DXbb;
        ljet.recoInclusion = 0;
        ljet.trueInclusion = 0;
        particles.push_back(ljet);
        ljetCandidates.push_back(ljet.p4);
    }
    nLjets_ll = ljetCandidates.size();
    if (ljetCandidates.size() == 0){
        return false;
    }
    // Process small-radius jets
    std::vector<TLorentzVector> sjetCandidates;
    for (size_t j = 0; j < jet_pt->size(); j++) {
        Particle sjet;
        sjet.p4.SetPtEtaPhiE(jet_pt->at(j), jet_eta->at(j), jet_phi->at(j), jet_e->at(j));
        sjet.type = 4;
        sjet.tagInfo = jet_DL1r->at(j);
        sjet.recoInclusion = 0;
        sjet.trueInclusion = 0;
        particles.push_back(sjet);
        sjetCandidates.push_back(sjet.p4);
    }
    // Fill particle vectors for this event
    // Get truth info TODO maybe use the same truth calculation as original method here?
    if (false) std::cout << "------------------------------------------" << std::endl; // Testing 15/01/25
    lepton_count = LowLevel_CountLeptons();
    truth_decay_mode = LowLevel_ClassifyDecayType();
    truth_decay_mode_old = LowLevel_ClassifyDecayType_OLD();
    if (false) std::cout << "Truth decay mode: " << truth_decay_mode << std::endl; // Testing 15/01/25
    if ((truth_decay_mode == 1) || (truth_decay_mode == 2)) successfulTruthMatch = LowLevel_MatchTruthParticles();
    std::tie(best_mH, best_mWH_qqbb, best_mWH_lvbb) = LowLevel_GetBestWhMasses();
    for (const auto &particle : particles) {
        ll_particle_px.push_back(particle.p4.Px());
        ll_particle_py.push_back(particle.p4.Py());
        ll_particle_pz.push_back(particle.p4.Pz());
        ll_particle_e.push_back(particle.p4.E());
        ll_particle_type.push_back(particle.type);
        ll_particle_tagInfo.push_back(particle.tagInfo);
        ll_particle_recoInclusion.push_back(particle.recoInclusion);
        ll_particle_trueInclusion.push_back(particle.trueInclusion);
    }
    // output_tree->Fill();
    if (debugMode) std::cout << "\t" << "Leaving LowLevel_Loop" << std::endl;
    return true;
}

void EventLoop::Loop()
{
    //assert(("You need to add the ttbar flavour filter tag to the output ntuples, but you were too lazy to do it earlier", false));
    //assert(("You also need to decide whether you want to have the normalisation factor for the H+Wh signal samples include the xsec from the PMG file, or just be normalised to 1. UPDATE I think Dominik just said go for 1", false));
    /* The main function which we call on each event. Loosely, this will do the following:
        - Reset variables to their 'default' values, to avoid as many errors as possible and so that if we accidentally (or on-purpose) ouptut 
            un-set variables, we know they are un-set and not just left from the previous event.
            This is the ResetVariables function.
        - Translate some of the input variables from the ntuples to the variables we'll use in the code (this step allows us to maintain mostly 
            the same code even if ntuples change). This might just be renaming, or it might be adding factors of 10^3, or it might be redefinining
            This is the TranslateVariables and CalcEventWeight functions.
        - Fill various containers with the relevant variables. In particular, we fill the lepton container, and jets (small and large R) containers
            with the information, and sort the jet vectors by Pt.
            This is the SetLeptonVectors and SetJetVectors functions.
        - Find the jets/lepton combinations which we think are most likely to be the candidates for the objects in our event (namely: a leptonically
            decaying W boson, a hadronically decaying W boson, and a Standard Model like 125GeV Higgs particle)
            This is acheived in the functions SetJetPair and GetWBoson, which use helper functions.
        - Check if the objects we have reconstructed pass some preselection cuts.
            This is achieved using the PassEventSelectionBoosted function, and the FindFJetPair function.
        - Get other event-level information.
            This is acheived using the functions: GetMwt, SetTruthParticles, btagCounting, FindTop, GetTruthMass
        - Write this information out (either to root histograms, csv or root ntuples)
            This is achieved using the functions: WriteEventOutStdout, WriteEventOutHist, WriteEventOutCsv, and the call output_tree->Fill().
    */
    
    if (debugMode) std::cout << "" << "Entering Loop" << std::endl;
    // std::cout << "Event number " << eventNumber << std::endl;
    //assert(("Really need to check the SetTruthParticles for loops, why is the dRmin being reset inside the for loop??? Seems problematic", false));
    //assert(("Need to check that the b-tag counting is done correctly", false));
    //assert(("Need to add more output variables to output-tree (eg. The Higgs mass, ptx, pty, ptz, etc.)", false));
    //assert(("Also probably worth putting in some 'isVariableSet' assert statements to check that the functions are called in the correct order (for future proofing)", false));

    // ---- Reset the persistent variables to their defaults -----
    ResetVariables();

    // ---- Translate variables from whatever form/names the input ntuples have given us into what use in the rest of the code -----
    bool passed_translation = TranslateVariables();
    if (!passed_translation){
        if (debugMode) std::cout << "\t" << "Event # " << eventNumber << " failed TranslateVariables" << std::endl;
        return;
    }
    if (debugMode) std::cout << "\t" << "Event # " << eventNumber << " passed TranslateVariables" << std::endl;
    
    // ---- Put lepton in a container -----
    bool passed_set_lep = SetLeptonVectors();
    if (!passed_set_lep){
        if (debugMode) std::cout << "\t" << "Event # " << eventNumber << " failed SetLeptonVectors" << std::endl;
        return;
    }
    if (debugMode) std::cout << "\t" << "Event # " << eventNumber << " passed SetLeptonVectors" << std::endl;

    // ---- Calculate the event weight -----
    CalcEventWeight();

    // ---- Fill vectors with jet information (calo jets, track jets, large-R jets. Doesn't do forward calo jets) -----
    SetJetVectors();

    // ---- Select the jets corresponding to Higgs and jet/lepton corresponding to W -----
    // Probably move SetJetPair to be called here
    // Key: Positively or a negatively charged lepton (LepN or LepP), a fat jet ("Merged"),  slimmer jets ("resolved"), background-enriched ("CR"), signal enriched ("SR").
    // This bool ensures that fat or slim jets are passed


    // ---- Check if this event passed the event selection -----
    bool passed_merged_preselection = PassEventSelectionBoosted(met_pt_min, lep_pt_min, higgs_pt_min, W_leptonic_pt_min, lep_SMHiggs_angle_min, lep_SMHiggs_angle_max, lep_W_hadronic_angle_min, hw_angle);
    bool passed_resovled_preselction = false; // My serch is only on the boosted channel
    if (!passed_merged_preselection && !passed_resovled_preselction)
    {
        if (debugMode) std::cout << "\t" << "Event # " << eventNumber << " failed EventSelectionBoosted" << std::endl;
        if (debugMode) std::cout << "\t" << "Failed EventSelectionBoosted and PassEventSelectionResolved" << std::endl;
        return;
    }
    if (debugMode) std::cout << "\t" << "Event # " << eventNumber << " passed EventSelectionBoosted" << std::endl;
    if (debugMode) std::cout << "\t" << "Passed EventSelectionBoosted or PassEventSelectionResolved" << std::endl;

    // ---- Get the transverse mass of the leptonic W -----
    m_mWT = GetMwt();

    // ---- Match the truth particles -----
    if (is_signal_sample) { // Should never have is_signal_sample without weights_included anyway.
        SetTruthParticles();
    }

    btagCounting();

    FindTop();

    /*
    // ---- Get some important variables we want to store -----
    if (Jets.size() >= 4 && Lepton_Charge < 0) // Why is this clause "(Jets.size() >= 4 && Lepton_Charge < 0)"? Doesn't matter too much for me atm as we don't use the output of MatchTruthParticlesToJets
    {
        if (is_signal_sample) MatchTruthParticlesToJets();
    }
    */
    
    if (weights_included) m_MassTruth = GetTruthMass(); // Think we don't want this to be done (or, at least, there is no point doing it), if we are running on data rather than MC. TODO confirm this.

    // ---- Do cutflow assignment for sideband region -----
    if (passed_merged_preselection)
    {
        if ((Higgs.M() * 0.001 < hmlb && hmlb_cut_on) || (Higgs.M() * 0.001 > hmub && hmub_cut_on))
        { // Failed the Higgs mass requirement, so don't throw away the event, but make it control region
            CutFlowAssignment(m_HiggsMassCutFlow, UnweightedCutFlow, WeightedCutFlow);
            if (debugMode) std::cout << "\t\t" << "Cutflow: Higgs mass requirement failed, putting event in control region" << std::endl;
            pass_sel["Merged_CR"] = true;
            if (is_signal_sample && truth_W_decay_mode == "qqbb") pass_sel["Merged_CR_subset_truth_jjbb"] = true;
            if (is_signal_sample && truth_W_decay_mode == "lvbb") pass_sel["Merged_CR_subset_truth_lvbb"] = true;
        }
        else
        { // Passed the Higgs mass requirement, so make it a Signal Region event
            pass_sel["Merged_SR"] = true;
            if (is_signal_sample && truth_W_decay_mode == "qqbb") pass_sel["Merged_SR_subset_truth_jjbb"] = true;
            if (is_signal_sample && truth_W_decay_mode == "lvbb") pass_sel["Merged_SR_subset_truth_lvbb"] = true;
            if (debugMode) std::cout << "\t\t" << "Cutflow: Higgs mass requirement passed, putting event in signal region" << std::endl;
        }
    }
    Set_Jet_observables();
    if (passed_resovled_preselction && !pass_sel["Merged_SR"])
    { // This bit will never be entered for the boosted channel
        if (Lepton_Charge > 0)
        {
            if (m_MaxMVA_Response > 0.7)
            {
                pass_sel["Resolved_LepP_SR"] = true;
                if (pass_sel["Merged_LepP_CR"] == true)
                    pass_sel["Merged_LepP_CR"] = false;
            }
            if (!pass_sel["Merged_LepP_CR"] && m_MaxMVA_Response > -0.5 && m_MaxMVA_Response < 0.3)
                pass_sel["Resolved_LepP_CR"] = true;
        }
        else if (Lepton_Charge < 0)
        {
            if (m_MaxMVA_Response > -2.0)
            {
                pass_sel["Resolved_LepN_SR"] = true;
                if (pass_sel["Merged_LepN_CR"] == true)
                    pass_sel["Merged_LepN_CR"] = false;
            }
            if (!pass_sel["Merged_LepN_CR"] && m_MaxMVA_Response < 0.5)
                pass_sel["Resolved_LepN_CR"] = true;
        }
    }

    // ---- Write the event output to histogram and maybe also to csv -----
    if (pass_sel["Merged_SR"] || pass_sel["Merged_CR"] || pass_sel["Resolved_LepN_CR"] || pass_sel["Resolved_LepP_CR"] || pass_sel["Resolved_LepN_SR"] || pass_sel["Resolved_LepP_SR"])
    {
        // fill the relevant variables
        nJets = Jets.size();
        nLJets = FJets.size();
        m_HT_bjets_Lepton_Pt = m_HT_bjets + (Leptons.at(0).Pt() * 0.001);
        m_pTH = Higgs.Pt() * 0.001;
        //m_pTH_over_mVH_qqbb = Higgs.Pt() * 0.001 / m_mVH_qqbb;
        //m_pTH_over_mVH_lvbb = Higgs.Pt() * 0.001 / m_mVH_lvbb;    
        m_mH = Higgs.M() * 0.001;
        m_mass_resolution_qqbb= (m_mVH_qqbb - m_MassTruth) / m_MassTruth;
        m_mass_resolution_lvbb= (m_mVH_lvbb - m_MassTruth) / m_MassTruth;
        m_MET_over_sqrtHT = (MET.Pt() * 0.001) / (std::sqrt(m_HT));
        m_pTW_leptonic = W_leptonic.Pt() * 0.001;
        m_mW_leptonic = W_leptonic.M() * 0.001;
        m_pTW_hadronic = W_hadronic.Pt() * 0.001;
        m_mW_hadronic = W_hadronic.M() * 0.001;
        m_LepEnergyFrac_qqbb = W_leptonic.Pt()/(Higgs.Pt()+W_leptonic.Pt()+W_hadronic.Pt());
        m_LepEnergyFrac_lvbb = W_leptonic.Pt()/(Higgs.Pt()+W_leptonic.Pt());
        m_deltaR_LH = fabs(Leptons.at(0).DeltaR(Higgs));
        m_deltaR_LWhad = fabs(Leptons.at(0).DeltaR(W_hadronic));
        m_deltaEta_HWhad = fabs(Higgs.Eta() - W_hadronic.Eta());
        m_deltaPhi_HWhad = fabs(Higgs.DeltaPhi(W_hadronic));
        m_deltaEta_HWlep = fabs(Higgs.Eta() - W_leptonic.Eta());
        m_deltaPhi_HWlep = fabs(Higgs.DeltaPhi(W_leptonic));
        ratio_Wpt_mVH_qqbb = W_hadronic.Pt()/m_mVH_qqbb;
        ratio_Wpt_mVH_lvbb = W_leptonic.Pt()/m_mVH_lvbb;
        ratio_Hpt_mVH_qqbb = Higgs.Pt()/m_mVH_qqbb;
        ratio_Hpt_mVH_lvbb = Higgs.Pt()/m_mVH_lvbb;
        m_diff_trk_calo_btags = m_NTags_trkJ - m_NTags_caloJ;
        m_diff_trk_calo_jets  = TrkJets.size() - Jets.size();

        m_H_mass        = Higgs.M() * 0.001;
        m_H_phi         = Higgs.Phi();
        m_H_eta         = Higgs.Eta();
        m_H_Pt          = Higgs.Pt() * 0.001;
        m_Whad_mass     = W_hadronic.M() * 0.001;
        m_Whad_phi      = W_hadronic.Phi();
        m_Whad_eta      = W_hadronic.Eta();
        m_Whad_Pt       = W_hadronic.Pt() * 0.001;
        m_Whad_FromSmallRJets_mass     = W_hadronic_FromSmallRJets.M() * 0.001;
        m_Whad_FromSmallRJets_phi      = W_hadronic_FromSmallRJets.Phi();
        m_Whad_FromSmallRJets_eta      = W_hadronic_FromSmallRJets.Eta();
        m_Whad_FromSmallRJets_Pt       = W_hadronic_FromSmallRJets.Pt() * 0.001;
        m_Whad_FromLargeRJet_mass     = W_hadronic_FromLargeRJet.M() * 0.001;
        m_Whad_FromLargeRJet_phi      = W_hadronic_FromLargeRJet.Phi();
        m_Whad_FromLargeRJet_eta      = W_hadronic_FromLargeRJet.Eta();
        m_Whad_FromLargeRJet_Pt       = W_hadronic_FromLargeRJet.Pt() * 0.001;
        m_Wlep_mass     = W_leptonic.M() * 0.001;
        m_Wlep_phi      = W_leptonic.Phi();
        m_Wlep_eta      = W_leptonic.Eta();
        m_Wlep_Pt       = W_leptonic.Pt() * 0.001;
        m_lep_mass      = Leptons.at(0).M() * 0.001;
        m_lep_phi       = Leptons.at(0).Phi();
        m_lep_eta       = Leptons.at(0).Eta();
        m_lep_Pt        = Leptons.at(0).Pt() * 0.001;
        
        // Optionally write to stdout (if we are in debug mode)
        if (debugMode) WriteEventOutStdout();
        //std::cout << "Event Number: " << eventNumber << ", EventWeight=" << EventWeight << std::endl;
    }
    Fill_NN_Scores();
    //WriteMVAInput();
    if (debugMode) std::cout << "" << "Leaving Loop" << std::endl;
}



bool EventLoop::TranslateVariables()
{
    if (debugMode) std::cout << "\t" << "Entering TranslateVariables" << std::endl;
    // This function translates the new ntuple input variables (ntuples used on 17/01/2022)
    // into the variables used in the code (which are those from the old ntuples)
    //EventWeight = 1; // TODO: Fill this in with the actual weight calculation
    MET.SetPtEtaPhiM(met_met, 0., met_phi, 0.); // TODO check this is correct/reasonable
    if ( (mu_charge->size() + el_charge->size()) != 1) {
        //throw std::runtime_error(std::string("Multiple leptons or no leptons found (ie, =/= 1 lepton)"));
        return false;
    }
    if (el_charge->size() == 1) {
        Lepton_Charge = el_charge->at(0);
    }
    else {
        Lepton_Charge = mu_charge->at(0);
    }
    if (debugMode) std::cout << "\t\t\t" << "Lepton_Charge = " << Lepton_Charge << std::endl;
    // Higgs_Truth will be sorted out by the function SetTruthParticles, which will be called by the main Loop just after this function ends
    // Same for Wplus_Truth
    FatJet_M = ljet_m;
    FatJet_PT = ljet_pt;
    FatJet_Eta = ljet_eta; // Unsure if used; may just be copied over but should include for now
    FatJet_Phi = ljet_phi; // Unsure if used; may just be copied over but should include for now
    
    //TLorentzVector JetTmp; // Temporary for getting the M values for the jets and trkJets. Probably a better way of doing this.
    signal_Jet_E = jet_e; // Used but maybe just for copying over (unsure at the moment); not sure what a 'signal jet' is; think this is just a particular choice of jet. 
    signal_Jet_PT = jet_pt; // Used but maybe just for copying over (unsure at the moment); not sure what a 'signal jet' is; think this is just a particular choice of jet. 
    signal_Jet_Eta = jet_eta; // Used but maybe just for copying over (unsure at the moment); not sure what a 'signal jet' is; think this is just a particular choice of jet. 
    signal_Jet_Phi = jet_phi; // Used but maybe just for copying over (unsure at the moment); not sure what a 'signal jet' is; think this is just a particular choice of jet. 
    //JetTmp.SetPtEtaPhiE(jet_pt, jet_eta, jet_phi, jet_e);
    //signal_Jet_M = JetTmp.M(); //---USED---
    TrackJet_PT = tjet_pt; //---USED---
    TrackJet_Eta = tjet_eta; //---USED---
    TrackJet_Phi = tjet_phi; //---USED---
    TrackJet_E = tjet_e; //---USED---
    //JetTmp.SetPtEtaPhiE(tjet_pt, tjet_eta, tjet_phi, tjet_e);
    //TrackJet_M = JetTmp.M(); //---USED---
    signal_Jet_tagWeightBin_DL1r_Continuous = jet_tagWeightBin_DL1r_Continuous;
    track_Jet_tagWeightBin_DL1r_Continuous = tjet_tagWeightBin_DL1r_Continuous;
    el_LHTight = el_isTight; // Is this correct? I copied this from Simon's (resolved channel) code.
    if (debugMode) std::cout << "\t" << "Leaving TranslateVariables" << std::endl;
    return true;
}

void EventLoop::ResetVariables()
{
    // Function to reset all the variables (especially important for those which may or may not be set in any given loop) to their dafault variables
    if (debugMode) std::cout << "\t" << "Entering ResetVariables" << std::endl;
    // Set all NN stuff to 0
    for (auto & kv_pair:nn_map_lvbb){
        lvbbNNs[kv_pair.first] = -1; // initialise to -1
        if ((doCombined) && (kv_pair.first.find("OddTrained") != std::string::npos)){
            basename_delim_pos = kv_pair.first.find_last_of("_");
            NN_basename = kv_pair.first.substr(0, basename_delim_pos);
            lvbbNNs[NN_basename + "_Combined"] = -1;
        }
    }
    for (auto & kv_pair:nn_map_qqbb){
        qqbbNNs[kv_pair.first] = -1; // initialise to -1
        if ((doCombined) && (kv_pair.first.find("OddTrained") != std::string::npos)){
            basename_delim_pos = kv_pair.first.find_last_of("_");
            NN_basename = kv_pair.first.substr(0, basename_delim_pos);
            qqbbNNs[NN_basename + "_Combined"] = -1;
        }
    }
    combined_category = -1;
    // Now other stuff
    for (auto sel : mySel) {
        pass_sel[sel] = false;
    }
    truth_W_decay_mode = "None";
    truth_lep_charge = 0;
    truth_agreement = 0;
    lep_charge_agreement = 0;
    m_NTags = -1;
    m_NTags_caloJ = -1;
    m_NTags_trkJ = -1;
    m_NTags_Higgs = -1;
    m_ntagsOutside = -1;
    m_MassTruth = -1;
    m_DeltaPhi_H_Lep = -1;
    m_DeltaPhi_H_MET = -1;
    m_DeltaPhi_W_hadronic_Lep = -1;
    m_DeltaPhi_W_hadronic_MET =-1;
    xbb_tag_higgsJet_value = 0;
    Xbb_variable_FJet_Higgs = -10;
    Xbb_variable_FJet_WHad = -10;
    selection_category = -1;
    m_EventWeights.clear();
    m_pTW_leptonic      = -1;
    m_mW_leptonic       = -1;
    m_pTW_hadronic      = -1;
    m_mW_hadronic       =-1;
    m_LepEnergyFrac_qqbb =-1;
    m_LepEnergyFrac_lvbb =-1;
    m_deltaR_LH         =-1;
    m_deltaR_LWhad      =-1;
    m_deltaEta_HWhad    =-1;
    m_deltaPhi_HWhad    =-1;
    m_deltaEta_HWlep    =-1;
    m_deltaPhi_HWlep    =-1;
    ratio_Wpt_mVH_qqbb  =-1;
    ratio_Wpt_mVH_lvbb  =-1;
    ratio_Hpt_mVH_qqbb  =-1;
    ratio_Hpt_mVH_lvbb  =-1;
    m_mTop_lepto = -1;
    m_mTop_hadro = -1;
    m_diff_trk_calo_btags = -10;
    m_diff_trk_calo_jets  = -10;
    Higgs_Truth.SetPtEtaPhiM(99999999,99999999,99999999,99999999);
    Wplus_Truth.SetPtEtaPhiM(99999999,99999999,99999999,99999999);
    index_W_jet_1=-1;
    index_W_jet_2=-1;
    index_H_jet_1=-1;
    index_H_jet_2=-1;
    m_nTrkTagsOutside=-1;
    m_nTrkTagsInW=-1;
    m_nTrkTagsInH=-1;
    m_nTrkTagsOutside_smallR=-1;
    m_nTrkTagsInW_smallR=-1;
    m_nTrkTagsInH_smallR=-1;
    m_nTrkTagsOutside_largeR=-1;
    m_nTrkTagsInW_largeR=-1;
    m_nTrkTagsInH_largeR=-1;
    Lepton_Pt=-999;
    Lepton_Eta=-999;
    Lepton_Phi=-999;
    Lepton_M=-999;
    Truth_Higgs_Pt=-999;
    Truth_Higgs_Eta=0;
    Truth_Higgs_Phi=-7;
    Truth_Higgs_M=-999;
    Truth_Wplus_Pt=-999;
    Truth_Wplus_Eta=0;
    Truth_Wplus_Phi=-7;
    Truth_Wplus_M=-999;

    nJets=-1;
    nLJets=-1;
    m_min_DeltaPhiJETMET=-1;
    selection_category = -1;
    EventWeight = 0;
    m_mH = -999;
    m_DeltaPhi_HW_hadronic = 999;
    m_DeltaR_HW_hadronic = 999;
    m_DeltaR_HW_leptonic = 999;
    m_mWT = -999;
    m_mVH_qqbb = -999;
    m_mVH_lvbb = -999;
    m_mVH_qqbb_WFromSmallRJets = -999;
    m_mVH_qqbb_WFromLargeRJet = -999;
    m_H_mass = -999;
    m_H_phi = -1;
    m_H_eta = 999;
    m_H_Pt = -999;
    m_Whad_mass = -999;
    m_Whad_phi = -1;
    m_Whad_eta = 999;
    m_Whad_Pt = -999;
    m_Whad_FromSmallRJets_mass = -999;
    m_Whad_FromSmallRJets_phi = -1;
    m_Whad_FromSmallRJets_eta = 999;
    m_Whad_FromSmallRJets_Pt = -999;
    m_Whad_FromLargeRJet_mass = -999;
    m_Whad_FromLargeRJet_phi = -1;
    m_Whad_FromLargeRJet_eta = 999;
    m_Whad_FromLargeRJet_Pt = -999;
    m_Wlep_mass = -999;
    m_Wlep_phi = -1;
    m_Wlep_eta = 999;
    m_Wlep_Pt = -999;
    m_bTagCategory = -1;
    m_mass_resolution_qqbb = -999;
    m_mass_resolution_lvbb = -999;
    // for (auto & sel : pass_sel){
    //     sel = false;
    // }
    pass_sel["Merged_CR"] = false;
    pass_sel["Merged_SR"] = false;
    m_mTop_best = -1;
    m_diff_trk_calo_btags=0;
    m_diff_trk_calo_jets=0;
    Jets_Pt.clear();
    Jets_Eta.clear();
    Jets_Phi.clear();
    Jets_M.clear();
    Jets_tagWeightBinDL1rContinuous.clear();
    FJets_Pt.clear();
    FJets_Eta.clear();
    FJets_Phi.clear();
    FJets_M.clear();
    FJets_DXbb.clear();
    
    if (debugMode) std::cout << "\t" << "Leaving ResetVariables" << std::endl;
}

bool EventLoop::WriteEventOutStdout()
{
    if (debugMode) std::cout << "\t\t" << "Entering WriteEventOutStdout" << std::endl;

    if (debugMode) std::cout << "\t\t\t" << "Filling TTree with the following values:" << std::endl;
    if (debugMode) std::cout << "\t\t\t" << "MET = " << MET.Pt() << std::endl;
    if (debugMode) std::cout << "\t\t\t" << "Lepton_Eta = " << Leptons.at(0).Eta() << std::endl;
    if (debugMode) std::cout << "\t\t\t" << "Lepton_Pt = " << Leptons.at(0).Pt() << std::endl;
    if (debugMode) std::cout << "\t\t\t" << "Lepton_Phi = " << Leptons.at(0).Phi() << std::endl;
    if (debugMode) std::cout << "\t\t\t" << "Lepton_M = " << Leptons.at(0).M() << std::endl;
    if (debugMode) std::cout << "\t\t\t" << "m_NTags = " << m_NTags << std::endl;
    if (debugMode) std::cout << "\t\t\t" << "Jets.size() = " << Jets.size() << std::endl;
    if (debugMode) std::cout << "\t\t\t" << "FJets.size() = " << FJets.size() << std::endl;
    if (debugMode) std::cout << "\t\t\t" << "m_mWT = " << m_mWT << std::endl;
    if (debugMode) std::cout << "\t\t\t" << "min_DeltaPhiJETMET = " << m_min_DeltaPhiJETMET << std::endl;
    if (debugMode) std::cout << "\t\t\t" << "DeltaPhi_H_Lep = " << m_DeltaPhi_H_Lep << std::endl;
    if (debugMode) std::cout << "\t\t\t" << "DeltaPhi_H_MET = " << m_DeltaPhi_H_MET << std::endl;
    if (debugMode) std::cout << "\t\t\t" << "DeltaPhi_W_hadronic_Lep = " << m_DeltaPhi_W_hadronic_Lep << std::endl;
    if (debugMode) std::cout << "\t\t\t" << "DeltaPhi_W_hadronic_MET = " << m_DeltaPhi_W_hadronic_MET << std::endl;
    if (debugMode) std::cout << "\t\t\t" << "HT = " << m_HT << std::endl;
    if (debugMode) std::cout << "\t\t\t" << "HT_bjets = " << m_HT_bjets << std::endl;
    if (debugMode) std::cout << "\t\t\t" << "HT_bjets_Lepton_Pt = " << m_HT_bjets_Lepton_Pt << std::endl;
    if (debugMode) std::cout << "\t\t\t" << "m_mVH_qqbb = " << m_mVH_qqbb << std::endl;
    if (debugMode && (m_mVH_qqbb < 0)) std::cout << "EventNumber: " << eventNumber << " has negative mVH_qqbb: " << m_mVH_qqbb << std::endl;
    if (debugMode) std::cout << "\t\t\t" << "m_mVH_lvbb = " << m_mVH_lvbb << std::endl;
    if (debugMode && (m_mVH_lvbb < 0)) std::cout << "EventNumber: " << eventNumber << " has negative mVH_lvbb: " << m_mVH_lvbb << std::endl;
    if (debugMode) std::cout << "\t\t\t" << "DeltaPhi_HW_hadronic = " << m_DeltaPhi_HW_hadronic << std::endl;
    if (debugMode) std::cout << "\t\t\t" << "DeltaPhi_HW_leptonic = " << m_DeltaPhi_HW_leptonic << std::endl;
    if (debugMode) std::cout << "\t\t\t" << "DeltaR_HW_hadronic = " << m_DeltaR_HW_hadronic << std::endl;
    if (debugMode) std::cout << "\t\t\t" << "DeltaR_HW_leptonic = " << m_DeltaR_HW_leptonic << std::endl;
    if (debugMode) std::cout << "\t\t\t" << "MaxMVA_Response = " << m_MaxMVA_Response << std::endl;
    if (debugMode) std::cout << "\t\t\t" << "pTH = " << m_pTH << std::endl;
    if (debugMode) std::cout << "\t\t\t" << "pTH_over_mVH_qqbb = " << m_pTH_over_mVH_qqbb << std::endl;
    if (debugMode) std::cout << "\t\t\t" << "pTH_over_mVH_lvbb = " << m_pTH_over_mVH_lvbb << std::endl;
    if (debugMode) std::cout << "\t\t\t" << "mH = " << m_mH << std::endl;
    if (debugMode) std::cout << "\t\t\t" << "bTagCategory = " << m_bTagCategory << std::endl;
    if (debugMode) std::cout << "\t\t\t" << "mass_resolution_qqbb = " << m_mass_resolution_qqbb << std::endl;
    if (debugMode) std::cout << "\t\t\t" << "mass_resolution_lvbb = " << m_mass_resolution_lvbb << std::endl;
    if (debugMode) std::cout << "\t\t\t" << "MET_over_sqrtHT = " << m_MET_over_sqrtHT << std::endl;
    if (debugMode) std::cout << "\t\t\t" << "NTags_trkJ = " << m_NTags_trkJ << std::endl;
    if (debugMode) std::cout << "\t\t\t" << "ljet_Xbb2020v3_Higgs = " << xbb_tag_higgsJet_value << std::endl;
    if (debugMode) std::cout << "\t\t\t" << "Xbb_variable_FJet_Higgs = " << Xbb_variable_FJet_Higgs << std::endl;
    if (debugMode) std::cout << "\t\t\t" << "Xbb_variable_FJet_WHad = " << Xbb_variable_FJet_WHad << std::endl;
    if (debugMode) std::cout << "\t\t\t" << "selection_category = " << selection_category << std::endl;
    if (debugMode) std::cout << "\t\t\t" << "DSID = " << dsid << std::endl;
    if (debugMode) std::cout << "\t\t\t" << "truth_W_decay_mode = " << truth_W_decay_mode << std::endl;
    if (debugMode) std::cout << "\t\t\t" << "truth_lep_charge = " << truth_lep_charge << std::endl;
    if (debugMode) std::cout << "\t\t\t" << "truth_agreement = " << truth_agreement << std::endl;
    if (debugMode) std::cout << "\t\t\t" << "lep_charge_agreement = " << lep_charge_agreement << std::endl;
    if (debugMode) std::cout << "\t\t\t" << "pTW_leptonic = " << m_pTW_leptonic << std::endl;
    if (debugMode) std::cout << "\t\t\t" << "mW_leptonic = " << m_mW_leptonic << std::endl;
    if (debugMode) std::cout << "\t\t\t" << "pTW_hadronic = " << m_pTW_hadronic << std::endl;
    if (debugMode) std::cout << "\t\t\t" << "mW_hadronic = " << m_mW_hadronic << std::endl;
    if (debugMode) std::cout << "\t\t\t" << "LepEnergyFrac_qqbb = " << m_LepEnergyFrac_qqbb << std::endl;
    if (debugMode) std::cout << "\t\t\t" << "LepEnergyFrac_lvbb = " << m_LepEnergyFrac_lvbb << std::endl;
    if (debugMode) std::cout << "\t\t\t" << "deltaR_LH = " << m_deltaR_LH << std::endl;
    if (debugMode) std::cout << "\t\t\t" << "deltaR_LWhad = " << m_deltaR_LWhad << std::endl;
    if (debugMode) std::cout << "\t\t\t" << "deltaEta_HWhad = " << m_deltaEta_HWhad << std::endl;
    if (debugMode) std::cout << "\t\t\t" << "deltaPhi_HWhad = " << m_deltaPhi_HWhad << std::endl;
    if (debugMode) std::cout << "\t\t\t" << "deltaEta_HWlep = " << m_deltaEta_HWlep << std::endl;
    if (debugMode) std::cout << "\t\t\t" << "deltaPhi_HWlep = " << m_deltaPhi_HWlep << std::endl;
    if (debugMode) std::cout << "\t\t\t" << "ratio_Wpt_mVH_qqbb = " << ratio_Wpt_mVH_qqbb << std::endl;
    if (debugMode) std::cout << "\t\t\t" << "ratio_Wpt_mVH_lvbb = " << ratio_Wpt_mVH_lvbb << std::endl;
    if (debugMode) std::cout << "\t\t\t" << "ratio_Hpt_mVH_qqbb = " << ratio_Hpt_mVH_qqbb << std::endl;
    if (debugMode) std::cout << "\t\t\t" << "ratio_Hpt_mVH_lvbb = " << ratio_Hpt_mVH_lvbb << std::endl;
    if (debugMode) std::cout << "\t\t\t" << "mTop_lepto = " << m_mTop_lepto << std::endl;
    if (debugMode) std::cout << "\t\t\t" << "mTop_hadro = " << m_mTop_hadro << std::endl;
    if (debugMode) std::cout << "\t\t\t" << "pass_Merged_CR = " << pass_sel["Merged_CR"] << std::endl;

    if (debugMode) std::cout << "\t\t" << "Leaving WriteEventOutStdout" << std::endl;
    return true;
}


void EventLoop::WriteTreeToFile(TFile *outFile)
{
    if (debugMode) std::cout << "\t" << "Entering WriteTreeToFile" << std::endl;
    outFile->cd();
    output_tree->Write();
    if (debugMode) std::cout << "\t" << "Leaving WriteTreeToFile" << std::endl;
}


void EventLoop::SetDebugMode(bool t_debugMode)
{
   if (t_debugMode) std::cout << "\t" << "Entering SetDebugMode" << std::endl;
   debugMode = t_debugMode;
}

void EventLoop::CalcEventWeight(){
    if (debugMode) std::cout << "\t" << "Entering CalcEventWeight" << std::endl;
    if (weights_included) {
        assert(("luminosity_factor must be set and > 0", luminosity_factor > 0));
        //float luminosity_factor = (randomRunNumber<=311481 ? 36207.66 : (randomRunNumber<=340453 ? 44307.4 : 58450.1));
        EventWeight = weight_normalise*weight_pileup*weight_mc*weight_leptonSF*weight_bTagSF_DL1r_Continuous*weight_jvt*luminosity_factor; ////*(ejets_2015_DL1r==1 || ejets_2016_DL1r==1 || ejets_2017_DL1r==1 || ejets_2018_DL1r==1 || mujets_2015_DL1r==1 || mujets_2016_DL1r==1 || mujets_2017_DL1r==1 || mujets_2018_DL1r==1);
        bool non_ttbar_filter_str, ttbar_filter_str;
        non_ttbar_filter_str = ((!(dsid==410470)) && (!(dsid==411073)) && (!(dsid==411074)) && (!(dsid==411075)) && (!(dsid==411076)) && (!(dsid==411077)) && (!(dsid==411078)) && (!(dsid==407342)) && (!(dsid==407343)) && (!(dsid==407344)) && (!(dsid==407348)) && (!(dsid==407349)));
        if (ttbarSelection == "Nominal"){
            ttbar_filter_str = (dsid==410470);
        } else if (ttbarSelection == "Bfilt"){
            ttbar_filter_str = (((dsid==410470) && (TopHeavyFlavorFilterFlag==0)) || ( ( dsid==411073 || dsid==411076) && TopHeavyFlavorFilterFlag==1) || ((dsid==411074 || dsid==411077) && TopHeavyFlavorFilterFlag==2) || ((dsid==411075 || dsid==411078) && TopHeavyFlavorFilterFlag==3));
        } else if (ttbarSelection == "Bfilt + HTfilt"){
            ttbar_filter_str = (((TopHeavyFlavorFilterFlag==0) && ( (dsid==410470 && (GenFiltHT<600000))  || (dsid==407344 && (GenFiltHT>600000 || GenFiltHT<1000000)) || (dsid==407343 && (GenFiltHT>1000000 || GenFiltHT<1500000)) || (dsid==407342 && (GenFiltHT>1500000))))  || ((dsid==411073 || dsid==411076) && (TopHeavyFlavorFilterFlag==1))  || ((dsid==411074 || dsid==411077) && (TopHeavyFlavorFilterFlag==2))  || ((dsid==411075 || dsid==411078) && (TopHeavyFlavorFilterFlag==3)));
        } else if (ttbarSelection == "HTfilt"){
            ttbar_filter_str = ((dsid==410470 && GenFiltHT<600000) || (dsid==407344 && GenFiltHT>600000 && GenFiltHT<1000000) || (dsid==407343 && GenFiltHT>1000000 && GenFiltHT<1500000) || (dsid==407342 && GenFiltHT>1500000));
        }
        else{
            assert((false));
        }
        if (non_ttbar_filter_str | ttbar_filter_str){
        } else{
            EventWeight = 0;
        }
        // // TODO URGENT Doesn't the below filtering mean we should be using a different sumOfWeights?
        // if(is_ttbar){
        //     if (!(((mcChannelNumber==410470) && (TopHeavyFlavorFilterFlag==0)) || ( ( mcChannelNumber==411073 || mcChannelNumber==411076) && TopHeavyFlavorFilterFlag==1) || ((mcChannelNumber==411074 || mcChannelNumber==411077) && TopHeavyFlavorFilterFlag==2) || ((mcChannelNumber==411075 || mcChannelNumber==411078) && TopHeavyFlavorFilterFlag==3)))
        //     {
        //         if (debugMode) std::cout << "\t\t" << "ttbar flavour filtered sample so setting event weight to 0" << std::endl;
        //         EventWeight=0;
        //     }
        // }
    } else {
        if (debugMode) std::cout << "\t\t" << "data so setting event weight to 1" << std::endl;
        EventWeight = 1; // In data mode, each event is just 1 event
    }
    std::cout.precision(10);
    if (debugMode) std::cout << "\t\t" << "Event Weight: " << EventWeight << std::endl;
    if (debugMode) std::cout << "\t\t" << "weight_normalise: " << weight_normalise << std::endl;
    m_EventWeights.push_back(EventWeight);
    if (debugMode) std::cout << "\t" << "Leaving CalcEventWeight" << std::endl;
}


void EventLoop::SetTruthParticles(){
   if (debugMode) std::cout << "\t" << "Entering SetTruthParticles" << std::endl;
   TLorentzVector truthParticle;
   bool use_status_23_quarks = false;
   bool use_status_52_quarks = false;
   bool contains_truth_Higgs = false;
   bool contains_truth_W     = false;
   int  n_attempts           = 0;
   found_Wplus_had_constituents           = false;
   found_Wplus_lep_constituents           = false;
   found_Higgs_constituents               = false;
   /*TopQuarks.clear();
   for(unsigned int i = 0; i< truth_pt->size(); i++){
       if(fabs(truth_pdgid->at(i)) == 6 && truth_status->at(i) == 62){  
             truthParticle.SetPtEtaPhiM(truth_pt->at(i),truth_eta->at(i),truth_phi->at(i),truth_m->at(i));
             TopQuarks.push_back(truthParticle);
       }
   }*/
   bool is_tau_case = false;
   while(!found_Higgs_constituents || !(found_Wplus_lep_constituents || found_Wplus_had_constituents)){
      bQuarks.clear();
      LightQuarks.clear();
      GenLevLeptons.clear();
      GenLevLeptonsPdgid.clear();
      
      if (debugMode) std::cout << "\t\t" << "Beginning loop over " << truth_pt->size() << " truth particles" << std::endl;
      for(unsigned int i = 0; i < truth_pt->size(); i++){
         if (debugMode) {
             std::cout << "\t\t\t" << "New truth particle, PDGID = " << std::setw (5) << truth_pdgid->at(i)
                                 << ", identified as " << std::setw (22) << pdgid_map[truth_pdgid->at(i)] 
                                 << ", m = " << std::setw (15) << truth_m->at(i) 
                                 << ", pt = " << std::setw (12) << truth_pt->at(i) 
                                 << ", eta = " << std::setw (13) << truth_eta->at(i) 
                                 << ", phi = " << std::setw (14) << truth_phi->at(i) 
                                 << std::endl;
             
         }
         ///std::cout<< truth_pdgid->at(i)<<"  "<<truth_status->at(i)<<"  "<<truth_barcode->at(i)<<"  "<<truth_pt->at(i)<<"  "<<truth_eta->at(i)<<std::endl;
         truthParticle.SetPtEtaPhiM(truth_pt->at(i),truth_eta->at(i),truth_phi->at(i),truth_m->at(i));
         if(fabs(truth_pdgid->at(i)) == 5 && (truth_status->at(i) == 71 || (use_status_23_quarks && truth_status->at(i) == 23))){
             bQuarks.push_back(truthParticle);
         }
         if(fabs(truth_pdgid->at(i)) <= 4 && (truth_status->at(i) == 71 || (use_status_52_quarks && (truth_status->at(i) == 52 || truth_status->at(i) == 51)))){
             LightQuarks.push_back(truthParticle);
         }
         if(fabs(truth_pdgid->at(i)) <= 16 && fabs(truth_pdgid->at(i)) >= 11 && (truth_status->at(i) == 1 || truth_status->at(i) == 2 )){
             GenLevLeptons.push_back(truthParticle);
             GenLevLeptonsPdgid.push_back(truth_pdgid->at(i));
         }
         if(fabs(truth_pdgid->at(i)) == 25){
             Higgs_Truth_P.SetPtEtaPhiM(truth_pt->at(i),truth_eta->at(i),truth_phi->at(i),truth_m->at(i));
             contains_truth_Higgs = true;
         }
         if(truth_pdgid->at(i) == 24){
             Wplus_Truth_P.SetPtEtaPhiM(truth_pt->at(i),truth_eta->at(i),truth_phi->at(i),truth_m->at(i));
             contains_truth_W     = true;
         }
      }
      
      float dRmin  = 42.0;
      for(unsigned int i = 0; i < bQuarks.size(); i++){
        //float dRmin  = 42.0;
         for(unsigned int j = i+1; j < bQuarks.size(); j++){
            if(i == j)continue;
            float dR = (bQuarks.at(i) + bQuarks.at(j)).DeltaR(Higgs_Truth_P);
            if(dR < 0.5 && dRmin > dR && contains_truth_Higgs){
                dRmin        = dR;
                Higgs_p1     = bQuarks.at(i);
                Higgs_p2     = bQuarks.at(j);
                found_Higgs_constituents  = true;
                if (debugMode) std::cout << "\t\t" << "New Higgs constituents with dR = " << dR << std::endl;
            }
         }
      }

      // Reset the dRmin variable because now we deal with W, not Higgs
      dRmin  = 42.0;
      for(unsigned int i = 0; i < LightQuarks.size(); i++){
         //float dRmin = 42.0; 
         for(unsigned int j = i+1; j < LightQuarks.size(); j++){
           if(i == j)continue;
           float dR = (LightQuarks.at(i) + LightQuarks.at(j)).DeltaR(Wplus_Truth_P);
           if(dR < 0.5 && dRmin > dR  && contains_truth_W){
                dRmin            = dR;
                Wplus_p1         = LightQuarks.at(i);
                Wplus_p2         = LightQuarks.at(j);
                found_Wplus_had_constituents  = true;
                is_tau_case = false;
                if (debugMode) std::cout << "\t\t" << "New Whad constituents with dR = " << dR << std::endl;
           }
         }
      }

      /*
      for(unsigned int i = 0; i < bQuarks.size(); i++){
         //float dRmin = 42.0; 
         for(unsigned int j = i+1; j < LightQuarks.size(); j++){
           float dR = (bQuarks.at(i) + LightQuarks.at(j)).DeltaR(Wplus_Truth_P);
           if(dR < 0.5 && dRmin > dR  && contains_truth_W){
                //dRmin            = dR;
                //Wplus_p1         = bQuarks.at(i);
                //Wplus_p2         = LightQuarks.at(j);
                //found_Wplus_had_constituents  = true;
                if (debugMode) std::cout << "\t\t" << "New MISSED Whad constituents with dR = " << dR << std::endl;
           }
         }
      }
      */
      // Keep same dRmin because we want to find best between qqbb/lvbb cases
      for(unsigned int i = 0; i < GenLevLeptons.size(); i++){
         for(unsigned int j = i+1; j < GenLevLeptons.size(); j++){
           if(i == j)continue;
           // Check this is a valiud pairing of lepton + antilepton neutrino (or vice versa)
           bool valid_pairing_e_or_mu = false;
           bool valid_pairing_tau = false;
           valid_pairing_e_or_mu = valid_pairing_e_or_mu || (GenLevLeptonsPdgid.at(i) == 11 && GenLevLeptonsPdgid.at(j) == -12);
           valid_pairing_e_or_mu = valid_pairing_e_or_mu || (GenLevLeptonsPdgid.at(i) == -11 && GenLevLeptonsPdgid.at(j) == 12);
           valid_pairing_e_or_mu = valid_pairing_e_or_mu || (GenLevLeptonsPdgid.at(i) == 12 && GenLevLeptonsPdgid.at(j) == -11);
           valid_pairing_e_or_mu = valid_pairing_e_or_mu || (GenLevLeptonsPdgid.at(i) == -12 && GenLevLeptonsPdgid.at(j) == 11);
           valid_pairing_e_or_mu = valid_pairing_e_or_mu || (GenLevLeptonsPdgid.at(i) == 13 && GenLevLeptonsPdgid.at(j) == -14);
           valid_pairing_e_or_mu = valid_pairing_e_or_mu || (GenLevLeptonsPdgid.at(i) == -13 && GenLevLeptonsPdgid.at(j) == 14);
           valid_pairing_e_or_mu = valid_pairing_e_or_mu || (GenLevLeptonsPdgid.at(i) == 14 && GenLevLeptonsPdgid.at(j) == -13);
           valid_pairing_e_or_mu = valid_pairing_e_or_mu || (GenLevLeptonsPdgid.at(i) == -14 && GenLevLeptonsPdgid.at(j) == 13);
           valid_pairing_tau = valid_pairing_tau || (GenLevLeptonsPdgid.at(i) == 15 && GenLevLeptonsPdgid.at(j) == -16);
           valid_pairing_tau = valid_pairing_tau || (GenLevLeptonsPdgid.at(i) == -15 && GenLevLeptonsPdgid.at(j) == 16);
           valid_pairing_tau = valid_pairing_tau || (GenLevLeptonsPdgid.at(i) == 16 && GenLevLeptonsPdgid.at(j) == -15);
           valid_pairing_tau = valid_pairing_tau || (GenLevLeptonsPdgid.at(i) == -16 && GenLevLeptonsPdgid.at(j) == 15);
           bool valid_pairing = valid_pairing_e_or_mu || valid_pairing_tau;
           if (!valid_pairing) continue;
           float dR = (GenLevLeptons.at(i) + GenLevLeptons.at(j)).DeltaR(Wplus_Truth_P);
           if(dR < 0.5 && dRmin > dR && contains_truth_W){
                dRmin            = dR;
                if (valid_pairing_e_or_mu) {
                    is_tau_case = false;
                    Wplus_p1         = GenLevLeptons.at(i);
                    Wplus_p2         = GenLevLeptons.at(j);
                    found_Wplus_lep_constituents  = true;
                    if (debugMode) std::cout << "\t\t" << "New Wlep constituents with dR = " << dR << std::endl;
                }
                if (valid_pairing_tau){
                    is_tau_case = true;
                    // If we have found a tau that matches better than a pair of jets, need to label this as a "NONE" channel event because we can't know the real truth
                    // So reset both the Wplus_lep and Wplus_had 'constituents found' bools to false
                    found_Wplus_lep_constituents  = false;
                    found_Wplus_had_constituents  = false;
                    if (debugMode) std::cout << "\t\t" << "New Wlep constituents with dR = " << dR << " but it's a tau so forbidding" << std::endl;
                }
           }
         }
      }

      if(!found_Higgs_constituents){
         use_status_23_quarks = true;
         if(n_attempts == 2)break;
         n_attempts++;
      }
      if(!(found_Wplus_lep_constituents || found_Wplus_had_constituents)){
         use_status_52_quarks = true;
         if(n_attempts == 2)break;
         n_attempts++;
      }
   }
   truth_W_decay_mode = (found_Higgs_constituents && found_Wplus_had_constituents) ? "qqbb" : ((found_Higgs_constituents && found_Wplus_lep_constituents) ? "lvbb" : "None");
   // We checked the Wplus_lep case second, which means that if we passed this then this must be a closer dR, so should take this (not sure what order the above line checks in, perhaps it's fine but I want to be explicit)
   truth_W_decay_mode = (found_Higgs_constituents && found_Wplus_lep_constituents) ? "lvbb" : truth_W_decay_mode;
   if (found_Higgs_constituents) {
    Higgs_Truth = (Higgs_p1 + Higgs_p2);
   }
   else{
    Higgs_Truth.SetPtEtaPhiM(99999999,99999999,99999999,99999999);
    if (debugMode) std::cout << "\t\t" << "Didn't find Higgs Truth" << std::endl;
   }
   if (found_Wplus_had_constituents || found_Wplus_lep_constituents) {
    Wplus_Truth = (Wplus_p1 + Wplus_p2);
   }
   else{
    Wplus_Truth.SetPtEtaPhiM(99999999,99999999,99999999,99999999);
    if (debugMode) std::cout << "\t\t" << "Didn't find Wplus Truth" << std::endl;
   }
    if (is_signal_sample) // Use truth information if available
    {
        if (truth_W_decay_mode == "qqbb") {
            if (debugMode) std::cout << "\t\t" << "Truth: Hadronic channel event" << std::endl;
        }
        else if (truth_W_decay_mode == "lvbb") {
            if (debugMode) std::cout << "\t\t" << "Truth: Leptonic channel event" << std::endl;
        }
        else {
            if (debugMode) std::cout << "\t\t" << "No truth particle found for signal event" << std::endl;
            //throw std::runtime_error(std::string("No truth matched for signal event"));;
        }
    } else {
        if (debugMode) std::cout << "\t\t" << "Non-signal event, so call this a 'None' channel event" << std::endl;
    }

    // Now do some checking for matching truth to reco lepton (this is all just for testing the truth matching)
    float dRmin = 42.0;
    for(unsigned int i = 0; i < GenLevLeptons.size(); i++){
        if (abs(GenLevLeptonsPdgid.at(i)) == 11 || abs(GenLevLeptonsPdgid.at(i)) == 13 || abs(GenLevLeptonsPdgid.at(i)) == 15){
            float dR = GenLevLeptons.at(i).DeltaR(Leptons.at(0));
            if(dRmin > dR){
                dRmin = dR;
                truth_lep_charge = (GenLevLeptonsPdgid.at(i) < 0)*1 + -1*(GenLevLeptonsPdgid.at(i) > 0);
                if (debugMode) std::cout << "\t\t" << "Updated truth lepton charge to: " << truth_lep_charge << " for particle: " << pdgid_map[GenLevLeptonsPdgid.at(i)] << " with new dRmin of: " << dRmin <<std::endl;
                if (debugMode) std::cout << "\t\t" << "Recall reco lepton charge: " << Lepton_Charge << "; Reco lepton Eta: " << Leptons.at(0).Eta() << "; Reco lepton Phi: " << Leptons.at(0).Phi() << "; Reco lepton flavour: " << ((el_charge->size() == 1) ? "El" : "Mu") << std::endl;
            }
        }
    }
    if (truth_lep_charge != 0) {
        if (truth_W_decay_mode != "None") {
            truth_agreement = ((truth_lep_charge > 0 && truth_W_decay_mode == "lvbb") || (truth_lep_charge < 0 && truth_W_decay_mode == "qqbb")) ? +1 : -1;
            if (debugMode) std::cout << "\t\t" << "Truth channel calculation agreement: " << truth_agreement << std::endl;
        }
        lep_charge_agreement = truth_lep_charge * Lepton_Charge;
        if (debugMode) std::cout << "\t\t" << "Truth/reco lepton charge agreement: " << lep_charge_agreement << std::endl;
    }
    if (is_tau_case) truth_agreement = -2;
    
    /*
    // Code to try and find the truth decay products of a truth anti-tau (if one exists) - this is just for some temporary debugging
    for(unsigned int i = 0; i < truth_pdgid->size(); i++){
        if (truth_pdgid->at(i) == -15){
            if (debugMode) std::cout << "\t\t" << "Running decay product search for anti-tau with Eta: " << std::setw (15) << truth_eta->at(i) << ", Phi: " << std::setw (15) << truth_phi->at(i) << std::endl;
            dRmin  = 42.0;
            for(unsigned int i = 0; i < LightQuarks.size(); i++){
                //float dRmin = 42.0; 
                for(unsigned int j = i+1; j < LightQuarks.size(); j++){
                    if(i == j)continue;
                    float dR = (LightQuarks.at(i) + LightQuarks.at(j)).DeltaR(Wplus_Truth_P);
                    if(dR < 0.5 && dRmin > dR  && contains_truth_W){
                            dRmin            = dR;
                            if (debugMode) std::cout << "\t\t" << "New anti-tau had constituents with dR = " << dR << std::endl;
                    }
                }
            }
            // Keep same dRmin because we want to find best between qqbb/lvbb cases
            for(unsigned int i = 0; i < GenLevLeptons.size(); i++){
                for(unsigned int j = i+1; j < GenLevLeptons.size(); j++){
                    if(i == j)continue;
                    // Check this is a valiud pairing of lepton + antilepton neutrino (or vice versa)
                    bool valid_pairing = false;
                    valid_pairing = valid_pairing || (GenLevLeptonsPdgid.at(i) == -11 && GenLevLeptonsPdgid.at(j) == 12);
                    valid_pairing = valid_pairing || (GenLevLeptonsPdgid.at(i) == 12 && GenLevLeptonsPdgid.at(j) == -11);
                    valid_pairing = valid_pairing || (GenLevLeptonsPdgid.at(i) == -13 && GenLevLeptonsPdgid.at(j) == 14);
                    valid_pairing = valid_pairing || (GenLevLeptonsPdgid.at(i) == 14 && GenLevLeptonsPdgid.at(j) == -13);
                    if (!valid_pairing) continue;
                    float dR = (GenLevLeptons.at(i) + GenLevLeptons.at(j)).DeltaR(Wplus_Truth_P);
                    if(dR < 0.5 && dRmin > dR && contains_truth_W){
                            dRmin            = dR;
                            if (debugMode) std::cout << "\t\t" << "New anti-tau lep constituents with dR = " << dR << std::endl;
                    }
                }
            }
        }
    }
    */
    // Truth_Higgs_Pt  = Higgs_Truth.Pt();
    // Truth_Higgs_Eta = Higgs_Truth.Eta();
    // Truth_Higgs_Phi = Higgs_Truth.Phi();
    // Truth_Higgs_M   = Higgs_Truth.M();
    // Truth_Wplus_Pt  = Wplus_Truth.Pt();
    // Truth_Wplus_Eta = Wplus_Truth.Eta();
    // Truth_Wplus_Phi = Wplus_Truth.Phi();
    // Truth_Wplus_M   = Wplus_Truth.M();

    Truth_Higgs_Pt  = Higgs_Truth_P.Pt();
    Truth_Higgs_Eta = Higgs_Truth_P.Eta();
    Truth_Higgs_Phi = Higgs_Truth_P.Phi();
    Truth_Higgs_M   = Higgs_Truth_P.M();
    Truth_Wplus_Pt  = Wplus_Truth_P.Pt();
    Truth_Wplus_Eta = Wplus_Truth_P.Eta();
    Truth_Wplus_Phi = Wplus_Truth_P.Phi();
    Truth_Wplus_M   = Wplus_Truth_P.M();
   
   if (debugMode) std::cout << "\t" << "Leaving SetTruthParticles" << std::endl;
}


int EventLoop::GetBTagCategoryShort(int NTags_InHiggsJet, int NTags_OutsideHiggsJet)
{
    if (debugMode) std::cout << "\t\t\t\t" << "Entering GetBTagCategoryShort" << std::endl;
    int category = -1;
    if (NTags_InHiggsJet == 0 && NTags_OutsideHiggsJet >= 2)
        category = 2;
    if (NTags_InHiggsJet == 1 && NTags_OutsideHiggsJet == 1)
        category = 2;
    if (NTags_InHiggsJet >= 2 && NTags_OutsideHiggsJet == 0)
        category = 2;
    if (NTags_InHiggsJet == 1 && NTags_OutsideHiggsJet >= 2)
        category = 3;
    if (NTags_InHiggsJet >= 2 && NTags_OutsideHiggsJet == 1)
        category = 3;
    if (NTags_InHiggsJet >= 2 && NTags_OutsideHiggsJet >= 2)
        category = 4;
    if (debugMode) std::cout << "\t\t\t\t\t" << "category = " << category << std::endl;
    if (debugMode) std::cout << "\t\t\t\t" << "Leaving GetBTagCategoryShort" << std::endl;
    return category;
}

int EventLoop::GetBTagCategory(int NTags_InHiggsJet, int NTags_OutsideHiggsJet)
{
    int category = -1;
    if (NTags_InHiggsJet == 0 && NTags_OutsideHiggsJet == 0)
        category = 0;
    if (NTags_InHiggsJet == 0 && NTags_OutsideHiggsJet == 1)
        category = 1;
    if (NTags_InHiggsJet == 0 && NTags_OutsideHiggsJet >= 2)
        category = 2;
    if (NTags_InHiggsJet == 1 && NTags_OutsideHiggsJet == 0)
        category = 3;
    if (NTags_InHiggsJet == 1 && NTags_OutsideHiggsJet == 1)
        category = 4;
    if (NTags_InHiggsJet == 1 && NTags_OutsideHiggsJet >= 2)
        category = 5;
    if (NTags_InHiggsJet >= 2 && NTags_OutsideHiggsJet == 0)
        category = 6;
    if (NTags_InHiggsJet >= 2 && NTags_OutsideHiggsJet == 1)
        category = 7;
    if (NTags_InHiggsJet >= 2 && NTags_OutsideHiggsJet >= 2)
        category = 8;
    return category;
}

double EventLoop::GetTruthMass()
{
    if (found_Higgs_constituents) // This used to be if (Higgs_Truth) to check if the pointer had been initialised, but Higgs_Truth is now a TLorentzVector rather than a TLorentzVector* (ie, no longer a pointer). TODO check if this is a sensible alternative.
    {
        return (Higgs_Truth + Wplus_Truth).M() * 0.001;
    }
    return 999;
}


bool EventLoop::FindFJetPair(Float_t higgs_pt_min, Float_t W_leptonic_pt_min, Float_t lep_SMHiggs_angle_min, Float_t lep_SMHiggs_angle_max, Float_t lep_W_hadronic_angle_min,
                             Float_t hw_angle)
{ //This function restricts the event based on the chosen jet parameters
    if (debugMode) std::cout << "\t\t" << "Entering FindFJetPair function" << std::endl;
    // bool status_W = false;
    
    m_DeltaPhi_HW_hadronic = fabs(W_hadronic.DeltaPhi(Higgs));
    m_DeltaPhi_HW_leptonic = fabs(W_leptonic.DeltaPhi(Higgs));
    m_DeltaR_HW_hadronic = fabs(W_hadronic.DeltaR(Higgs));
    m_DeltaR_HW_leptonic = fabs(W_leptonic.DeltaR(Higgs));
    m_DeltaPhi_H_Lep = fabs(Higgs.DeltaPhi(Leptons.at(0)));
    m_DeltaPhi_H_MET = fabs(Higgs.DeltaPhi(MET));
    m_DeltaPhi_W_hadronic_Lep = fabs(W_hadronic.DeltaPhi(Leptons.at(0)));
    m_DeltaPhi_W_hadronic_MET = fabs(W_hadronic.DeltaPhi(MET));
    
    if (Higgs.Pt() < higgs_pt_min && higgs_pt_min_cut_on)
    {
        if (debugMode) std::cout << "\t\t\t" << "Cutflow: HiggsPtCutFlow failed, discarding event. Higgs Pt = " << Higgs.Pt() << std::endl;
        CutFlowAssignment(m_HiggsPtCutFlow, UnweightedCutFlow, WeightedCutFlow);
        return false;
    }
    if (Higgs.DeltaR(Leptons.at(0)) < lep_SMHiggs_angle_min && lep_SMHiggs_angle_min_cut_on)
    {
        if (debugMode) std::cout << "\t\t\t" << "Cutflow: Higgs_LeptonAngleCutflow failed, discarding event. Higgs.DeltaR(Leptons.at(0)) = " << Higgs.DeltaR(Leptons.at(0)) << std::endl;
        CutFlowAssignment(m_Higgs_LeptonAngleCutflow, UnweightedCutFlow, WeightedCutFlow);
        return false;
    }
    if (Higgs.DeltaR(Leptons.at(0)) > lep_SMHiggs_angle_max && lep_SMHiggs_angle_max_cut_on)
    {
        if (debugMode) std::cout << "\t\t\t" << "Cutflow: Higgs_MaxLeptonAngleCutflow failed, discarding event. Higgs.DeltaR(Leptons.at(0)) = " << Higgs.DeltaR(Leptons.at(0)) << std::endl;
        CutFlowAssignment(m_Higgs_MaxLeptonAngleCutflow, UnweightedCutFlow, WeightedCutFlow);
        return false;
    }
    if (W_hadronic.DeltaR(Leptons.at(0)) < lep_W_hadronic_angle_min && lep_W_hadronic_angle_min_cut_on)
    {
        if (debugMode) std::cout << "\t\t\t" << "Cutflow: W_hadronic_LeptonAngleCutflow failed, discarding event. W_hadronic.DeltaR(Leptons.at(0)) = " << W_hadronic.DeltaR(Leptons.at(0)) << std::endl;
        CutFlowAssignment(m_W_hadronic_LeptonAngleCutflow, UnweightedCutFlow, WeightedCutFlow);
        return false;
    }
    if (m_DeltaPhi_HW_hadronic < hw_angle && hw_angle_cut_on)
    {
        if (debugMode) std::cout << "\t\t\t" << "Cutflow: Higgs_Whadronic_AngleCutflow failed, discarding event. DeltaPhi_HW_hadronic = " << m_DeltaPhi_HW_hadronic << std::endl;
        CutFlowAssignment(m_Higgs_WplusAngleCutflow, UnweightedCutFlow, WeightedCutFlow);
        return false;
    }
    /*
    if (debugMode) std::cout << "\t\t\t" << "Updating Wplus variable (using GetWBoson, since #FJets=1)" << std::endl;
    Wplus = GetWBoson(status_W); // Must be done here as status_W has scope limitations
    if ((!status_W)  && status_W_cut_on)
    {
        if (debugMode) std::cout << "\t\t\t" << "Cutflow: PositiveLepWCutflow failed, discarding event. status_W = " << status_W << std::endl;
        CutFlowAssignment(m_PositiveLepWCutflow, UnweightedCutFlow, WeightedCutFlow);
        return false;
    }
    */


    m_mVH_qqbb = (W_hadronic + Higgs).M() * 0.001;
    m_mVH_lvbb = (W_leptonic + Higgs).M() * 0.001;
    m_mVH_qqbb_WFromSmallRJets = (W_hadronic_FromSmallRJets + Higgs).M() * 0.001;
    m_mVH_qqbb_WFromLargeRJet = (W_hadronic_FromLargeRJet + Higgs).M() * 0.001;
    if (debugMode) std::cout << "\t\t" << "Leaving FindFJetPair function" << std::endl;
    return true;
}

void EventLoop::SetJetPair()
{ //THIS FUNCTION SETS THE SM HIGGS AND W BOSON JETS
    /*
    Function to determine which of the large-R Jets (ie, fat jets) and small jets from the input ntuple info
    we consider to be caused by the charged higgs and the W bosons from our decay event.
    Must have called SetJetVectors beforehand in order to have filled the vectors.
    */
    if (debugMode) std::cout << "\t\t" << "Entering SetJetPair" << std::endl;
    float DxbbThreshold = 2.44;
    float DXbb;
    //m_NTags = m_NTags_trkJ+m_NTags_caloJ; // Set as default, then update as appropriate later
    m_NTags = m_NTags_trkJ;
    Higgs = TLorentzVector(10000000.0, 10000000.0, 10000000.0, 10000000.0); // Impossibly big so it gets reassigned
    //if (debugMode) std::cout << "\t\t\t" << "Updating Wplus variable (Resetting, since #FJets>1)" << std::endl;
    W_hadronic = TLorentzVector(10000000.0, 10000000.0, 10000000.0, 10000000.0); // Hopefully reset later
    W_hadronic_FromSmallRJets = TLorentzVector(10000000.0, 10000000.0, 10000000.0, 10000000.0); // Hopefully reset later
    W_hadronic_FromLargeRJet = TLorentzVector(10000000.0, 10000000.0, 10000000.0, 10000000.0); // Hopefully reset later

    if (FJets.size() < 1) // Case where we have no fat jets; this is ignored for boosted channel
    {
        // TODO for this to be able to separate using qqbb/lvbb truth info, have to have called SetTruthParticles first, but at the moment this is called before that
        if (debugMode) std::cout << "\t\t\t" << "Number of FJets < 1" << std::endl;
        m_noJets.unweighted_vals["both_channels"]["twoTags"]++; // Just increment the twoTags, to keep track of these. This isn't a proper 'cutflow variable' but just making use of that structure to keep track of this.
        m_noJets.weighted_vals["both_channels"]["twoTags"] = m_noJets.weighted_vals["both_channels"]["twoTags"] + std::abs(EventWeight);
        if (is_signal_sample && truth_W_decay_mode == "qqbb")
        {
            m_noJets.unweighted_vals["jjbb_channel"]["twoTags"]++;
            m_noJets.weighted_vals["jjbb_channel"]["twoTags"] = m_noJets.weighted_vals["jjbb_channel"]["twoTags"] + std::abs(EventWeight);
        }
        if (is_signal_sample && truth_W_decay_mode == "lvbb")
        {
            m_noJets.unweighted_vals["lvbb_channel"]["twoTags"]++;
            m_noJets.weighted_vals["lvbb_channel"]["twoTags"] = m_noJets.weighted_vals["lvbb_channel"]["twoTags"] + std::abs(EventWeight);
        };
        m_NTags = m_NTags_trkJ;
        if (debugMode) std::cout << "\t\t" << "Leaving SetJetPair" << std::endl;
        return;
    }
    
    const double_t higgsMass = 125100; //TODO maybe this should be a constant defined at the top so it's not hidden away somewhere?
    const double_t wplusMass = 80379; //TODO maybe this should be a constant defined at the top so it's not hidden away somewhere?
    
    // Now loop over cases. Categories as follows
    // selection_category=0         One large-R jet, is Xbb tagged as H->bb. This is used as SMhiggs, Whad is reconstructed using GetWFromJets
    // selection_category=1         One large-R jet, not Xbb tagged as H->bb. The large-R jet mass is closer to mW than to mH, so we use this as Whad. SMhiggs is reconstructed using GetHFromJets
    // selection_category=2         One large-R jet, not Xbb tagged as H->bb. The large-R jet mass is closer to mH than to mW, so we use this as SMhiggs. WHad is reconstructed using GetWFromJets
    // selection_category=3         2+ large-R jets, exactly 1 tagged as H->bb. This is used as SMhiggs. In addition, the (different) large-R jet with mass closest to the Wmass passes a mass cut (mWhad < 110GeV) and passes a Pt cut (PtWhad> PtWlep)
    // selection_category=8         2+ large-R jets, exactly 1 tagged as H->bb. This is used as SMhiggs. In addition, the (different) large-R jet with mass closest to the Wmass fails  a mass cut (mWhad < 110GeV) and passes a Pt cut (PtWhad> PtWlep)
    // selection_category=9         2+ large-R jets, exactly 1 tagged as H->bb. This is used as SMhiggs. In addition, the (different) large-R jet with mass closest to the Wmass passes a mass cut (mWhad < 110GeV) and fails  a Pt cut (PtWhad> PtWlep)
    // selection_category=10        2+ large-R jets, exactly 1 tagged as H->bb. This is used as SMhiggs. In addition, the (different) large-R jet with mass closest to the Wmass fails  a mass cut (mWhad < 110GeV) and fails  a Pt cut (PtWhad> PtWlep)
    // selection_category=4         2+ large-R jets, more than 1 tagged as H->bb. We set the one with mass closest to true Higgs mass to be Higgs, then the (different) one with mass closest to the true W mass as a W.
    // selection_category=5         2+ large-R jets, none tagged as H->bb. We loop through large jets and for each check if it's closest to Higgs or W mass, then whichever is closer we try to update the best candidate if this jet is closer than the prev candidate. Both Higgs and W are set as large-R jets this way
    // selection_category=6         2+ large-R jets, none tagged as H->bb. We loop through large jets and for each check if it's closest to Higgs or W mass, then whichever is closer we try to update the best candidate if this jet is closer than the prev candidate. Higgs is set this way, but Whadronic is not and so is found with small jets
    // selection_category=7         2+ large-R jets, none tagged as H->bb. We loop through large jets and for each check if it's closest to Higgs or W mass, then whichever is closer we try to update the best candidate if this jet is closer than the prev candidate. Whadronic is set this way, but Higgs is not and so is found with small jets
    // selection_category=11        2+ large-R jets, none tagged as H->bb. We loop through large jets and for each check if it's closest to Higgs or W mass, then whichever is closer we try to update the best candidate if this jet is closer than the prev candidate. 

    if (FJets.size() == 1) // Case where we have exactly 1 fat jet
    {
        if (debugMode) std::cout << "\t\t\t" << "Number of FJets = 1" << std::endl;
        xbb_tag_higgsJet_value = ljet_Xbb2020v3_Higgs_REARRANGED.at(0);
        DXbb = log(ljet_Xbb2020v3_Higgs_REARRANGED.at(0)/(0.25*ljet_Xbb2020v3_Top_REARRANGED.at(0) + 0.75*ljet_Xbb2020v3_QCD_REARRANGED.at(0)));
        if (DXbb >= DxbbThreshold)
        {
            Higgs = FJets.at(0);
            // Apply small-R jet removal with respect to Higgs large-R jet
            ApplySmallRjetRemoval(&Jets, &JetIsTaggedBin, Higgs);
            W_hadronic = GetWFromJets();
            m_NTags = m_NTags_trkJ+m_NTags_caloJ; // no longer used
            //m_ntagsOutside = m_NTags_trkJ - nTaggedVRTrkJetsInFJet.at(0);
            m_bTagCategory = GetBTagCategory(m_NTags_Higgs, m_ntagsOutside);
            Xbb_variable_FJet_Higgs = DXbb;
            selection_category = 0;
            W_hadronic_FromSmallRJets = GetWFromJets();
        }
        else{
            if (abs( FJets.at(0).M() - wplusMass) < abs( FJets.at(0).M() - higgsMass))
            {
                W_hadronic = FJets.at(0);
                // Apply small-R jet removal with respect to W boson large-R jet
                ApplySmallRjetRemoval(&Jets, &JetIsTaggedBin, W_hadronic);
                Higgs = GetHFromJets();
                //m_ntagsOutside = m_NTags - nTaggedVRTrkJetsInFJet.at(0) - N_bInH;
                m_bTagCategory = GetBTagCategory(m_NTags_Higgs, m_ntagsOutside);
                selection_category = 2;
                W_hadronic_FromSmallRJets = GetWFromJets();
            }
            else
            {
                Higgs = FJets.at(0);
                // Apply small-R jet removal with respect to Higgs large-R jet
                ApplySmallRjetRemoval(&Jets, &JetIsTaggedBin, Higgs);
                W_hadronic = GetWFromJets();
                m_NTags = m_NTags_trkJ;
                //m_ntagsOutside = m_NTags_trkJ - nTaggedVRTrkJetsInFJet.at(0) - N_bInW;
                m_bTagCategory = GetBTagCategory(m_NTags_Higgs, m_ntagsOutside);
                selection_category = 1;
                W_hadronic_FromSmallRJets = GetWFromJets();

            }
        }
        if (debugMode) std::cout << "\t\t\t" << "m_NTags = " << m_NTags_trkJ << std::endl;
        if (debugMode) std::cout << "\t\t\t" << "m_ntagsOutside = " << m_ntagsOutside << std::endl;
        if (debugMode) std::cout << "\t\t\t" << "m_bTagCategory = " << m_bTagCategory << std::endl;
        if (debugMode) std::cout << "\t\t" << "Leaving SetJetPair" << std::endl;
        return;
    }

    // EVERYTHING FROM HERE AND BELOW IS WHERE WE HAVE > 1 FAT JET

    if (debugMode) std::cout << "\t\t\t" << "Number of FJets > 1" << std::endl;
    int indexWhadronic = -1;
    int indexHiggs = -1;
    int nFjetsTagged = 0;
    std::vector<int> TaggedFJetsIndices; // TODO will it make code faster if we have this as persistent and clear it, instead of remaking it each time?
    TaggedFJetsIndices.clear();
    for (int i = 0; i < FJets.size(); i++)
    {
        DXbb = log(ljet_Xbb2020v3_Higgs_REARRANGED.at(i)/(0.25*ljet_Xbb2020v3_Top_REARRANGED.at(i) + 0.75*ljet_Xbb2020v3_QCD_REARRANGED.at(i)));
        if (DXbb > DxbbThreshold)
        {
            nFjetsTagged++;
            TaggedFJetsIndices.push_back(i);
        }
    }
    if (nFjetsTagged == 1) // More than 1 fat jet, exactly one of them is tagged
    {
        if (debugMode) std::cout << "\t\t\t" << "Number of Tagged FJets = 1" << std::endl;
        indexHiggs = TaggedFJetsIndices.at(0);
        Xbb_variable_FJet_Higgs = log(ljet_Xbb2020v3_Higgs_REARRANGED.at(indexHiggs)/(0.25*ljet_Xbb2020v3_Top_REARRANGED.at(indexHiggs) + 0.75*ljet_Xbb2020v3_QCD_REARRANGED.at(indexHiggs)));
        Higgs = FJets.at(indexHiggs);
        // Apply small-R jet removal with respect to Higgs large-R jet
        ApplySmallRjetRemoval(&Jets, &JetIsTaggedBin, Higgs);
        if (debugMode) std::cout << "\t\t\t" << "Looping over " << FJets.size() << "large-R jets" << std::endl;
        bool bestW_set = false; // Checks if the large-R jet with mass most closesly matching the Wmass has been found (this should always be the case because we set the original one to be ridiculous)
        bool bestW_pass_mass_cut = false; // Checks if the large-R jet with mass most closesly matching the Wmass passes the mass cut
        bool bestW_pass_pt_cut = false; // Checks if the large-R jet with mass most closesly matching the Wmass passes the Pt cut
        for (int i = 0; i < FJets.size(); i++)
	    {
            if (debugMode) std::cout << "\t\t\t\t" << "large-R jet mass = " << FJets.at(i).M() << std::endl;
            bool Wokay = true;
            Wokay = Wokay && (i!=indexHiggs);
            if (debugMode && i==indexHiggs) std::cout << "\t\t\t\t" << "this jet is the Higgs tagged jet" << std::endl;
            //Wokay = Wokay && (abs(wplusMass - FJets.at(i).M()) < abs(wplusMass - W_hadronic.M()));
            //Wokay = Wokay && (FJets.at(i).Pt() > W_hadronic.Pt());
            //std::cout << "Event # " << eventNumber << "WhadPt = " << W_hadronic.Pt() << ", FJetPt = " << FJets.at(i).Pt() << std::endl;
            if(Wokay)
            {
                if (!bestW_set){
                    bestW_set = true; // Make sure that this happens at least once
                    bestW_pass_mass_cut = (FJets.at(i).M() < 110e3) && (FJets.at(i).M() > 50e3);
                    bestW_pass_pt_cut = FJets.at(i).Pt() > W_leptonic.Pt(); // Requires that Wleptonic has been set!
                    W_hadronic = FJets.at(i);
                    indexWhadronic = i;
                    if (debugMode) std::cout << "\t\t\t\t" << "Updated index Whadronic to: " << indexWhadronic << std::endl;
                }
                // If we have previously set another jet, but it DIDN'T pass the mass criteria, then we update if this one does
                if ((!bestW_pass_mass_cut) && ((FJets.at(i).M() < 110e3) && (FJets.at(i).M() > 50e3))){
                    bestW_pass_mass_cut = (FJets.at(i).M() < 110e3) && (FJets.at(i).M() > 50e3);
                    bestW_pass_pt_cut = FJets.at(i).Pt() > W_leptonic.Pt(); // Requires that Wleptonic has been set!
                    W_hadronic = FJets.at(i);
                    indexWhadronic = i;
                    if (debugMode) std::cout << "\t\t\t\t" << "Updated index Whadronic to: " << indexWhadronic << std::endl;
                }
                // Now check if it's actually the best, and if so update:
                if (((FJets.at(i).M() < 110e3) && (FJets.at(i).M() > 50e3)) && (FJets.at(i).Pt() > W_hadronic.Pt())){
                    bestW_pass_mass_cut = (FJets.at(i).M() < 110e3) && (FJets.at(i).M() > 50e3);
                    bestW_pass_pt_cut = FJets.at(i).Pt() > W_leptonic.Pt(); // Requires that Wleptonic has been set!
                    W_hadronic = FJets.at(i);
                    indexWhadronic = i;
                    if (debugMode) std::cout << "\t\t\t\t" << "Updated index Whadronic to: " << indexWhadronic << std::endl;
                }
            }
	    }
        if (bestW_set) {
            if (bestW_pass_mass_cut && bestW_pass_pt_cut) {
                Xbb_variable_FJet_WHad = log(ljet_Xbb2020v3_Higgs_REARRANGED.at(indexWhadronic)/(0.25*ljet_Xbb2020v3_Top_REARRANGED.at(indexWhadronic) + 0.75*ljet_Xbb2020v3_QCD_REARRANGED.at(indexWhadronic)));
                selection_category = 3;
                if (debugMode) std::cout << "\t\t\t\t" << "Set selection_category: " << selection_category << std::endl;
                W_hadronic_FromSmallRJets = GetWFromJets();
                W_hadronic_FromLargeRJet = FJets.at(indexWhadronic);
                // Apply small-R jet removal with respect to W large-R jet
                ApplySmallRjetRemoval(&Jets, &JetIsTaggedBin, W_hadronic);
            }
            else if ((!bestW_pass_mass_cut) && (bestW_pass_pt_cut)){
                selection_category = 8;
                if (debugMode) std::cout << "\t\t\t\t" << "Set selection_category: " << selection_category << std::endl;
                W_hadronic_FromSmallRJets = GetWFromJets();
                W_hadronic_FromLargeRJet = FJets.at(indexWhadronic);
                // This failed the mass cut on large-R jet W candidate, so revert this to having the W be from small-R jets
                W_hadronic = GetWFromJets();
            }
            else if ((bestW_pass_mass_cut) && (!bestW_pass_pt_cut)){
                selection_category = 9;
                if (debugMode) std::cout << "\t\t\t\t" << "Set selection_category: " << selection_category << std::endl;
                W_hadronic_FromSmallRJets = GetWFromJets();
                W_hadronic_FromLargeRJet = FJets.at(indexWhadronic);
                // For now, leave the W_hadronic, which will be used to make most of the other variables, as what it was before (ie the non-Higgs large-R jet with mass closest to the Wmass)
                // Also, for consistency, apply small-R jet removal with respect to W boson large-R jet
                ApplySmallRjetRemoval(&Jets, &JetIsTaggedBin, W_hadronic);
            }
            else if ((!bestW_pass_mass_cut) && (!bestW_pass_pt_cut)){
                selection_category = 10;
                if (debugMode) std::cout << "\t\t\t\t" << "Set selection_category: " << selection_category << std::endl;
                W_hadronic_FromSmallRJets = GetWFromJets();
                W_hadronic_FromLargeRJet = FJets.at(indexWhadronic);
                // This failed the mass cut on large-R jet W candidate, so revert this to having the W be from small-R jets
                W_hadronic = GetWFromJets();
            }
            //std::cout << "Event number: " << eventNumber << ";\t\tSelection category: " << selection_category << std::endl;
            //std::cout << "\t\tWhadronic selected Pt, Eta, Phi, M = (" << W_hadronic.Pt() << ", " << W_hadronic.Eta() << ", " << W_hadronic.Phi() << ", " << W_hadronic.M() << ")" << std::endl;
            //std::cout << "\t\tW_hadronic_FromSmallRJets large-R jet Pt, Eta, Phi, M = (" << W_hadronic_FromSmallRJets.Pt() << ", " << W_hadronic_FromSmallRJets.Eta() << ", " << W_hadronic_FromSmallRJets.Phi() << ", " << W_hadronic_FromSmallRJets.M() << ")" << std::endl;
            //std::cout << "\t\tW_hadronic_FromLargeRJet large-R jet Pt, Eta, Phi, M = (" << W_hadronic_FromLargeRJet.Pt() << ", " << W_hadronic_FromLargeRJet.Eta() << ", " << W_hadronic_FromLargeRJet.Phi() << ", " << W_hadronic_FromLargeRJet.M() << ")" << std::endl;
        }
        else {
            // Ideally we'd never get here
            if (debugMode) std::cout << "\t\t\t\t" << "Somehow we got to a place where we have 2+ large-R jets, one h->bb tagged, but none of the others close enough to Wmass. This isn't ideal. I'll set to be an error category (I haven't checked that this is particularly safe but I'll do it anyway)" << std::endl;
            selection_category = -1;
            if (debugMode) std::cout << "\t\t\t\t" << "Set selection_category: " << selection_category << std::endl;
            assert(("See above lines", false));
        }
    }
    else if (nFjetsTagged > 1) // More than 1 fat jet, more than one of them is tagged
    {
      Higgs.SetPtEtaPhiM(1000000000000,1,1,100000000);
      W_hadronic.SetPtEtaPhiM(1000000000000,1,1,100000000);
	  for( int k : TaggedFJetsIndices)
      // Have to reset the defaults here becuase we use mass, rather than Pt, to compare
        {
            if (abs(higgsMass - FJets.at(k).M() ) < abs(higgsMass - Higgs.M()))
            {
            Higgs = FJets.at(k);
            indexHiggs =k;
            }
        }
	  for (int i = 0; i < FJets.size(); i++)
	    { 
	      if(i!=indexHiggs &&	abs(wplusMass - FJets.at(i).M()) < abs(wplusMass - W_hadronic.M()) )
            {
            W_hadronic = FJets.at(i);
            indexWhadronic = i;
            }
	    }
        // Apply small-R jet removal with respect to Higgs large-R jet
        ApplySmallRjetRemoval(&Jets, &JetIsTaggedBin, Higgs);
        // Apply small-R jet removal with respect to W boson large-R jet
        ApplySmallRjetRemoval(&Jets, &JetIsTaggedBin, W_hadronic);
        selection_category = 4;
        if (debugMode) std::cout << "\t\t\t\t" << "Set selection_category: " << selection_category << std::endl;
        Xbb_variable_FJet_Higgs = log(ljet_Xbb2020v3_Higgs_REARRANGED.at(indexHiggs)/(0.25*ljet_Xbb2020v3_Top_REARRANGED.at(indexHiggs) + 0.75*ljet_Xbb2020v3_QCD_REARRANGED.at(indexHiggs)));
        Xbb_variable_FJet_WHad = log(ljet_Xbb2020v3_Higgs_REARRANGED.at(indexWhadronic)/(0.25*ljet_Xbb2020v3_Top_REARRANGED.at(indexWhadronic) + 0.75*ljet_Xbb2020v3_QCD_REARRANGED.at(indexWhadronic)));
        if (debugMode) std::cout << "\t\t\t\t" << "JANKY1" << std::endl;
    }
    else // more than 1 fat jet, none of them tagged
    {
        TLorentzVector currHiggs = TLorentzVector(10000000.0, 10000000.0, 10000000.0, 10000000.0);
        TLorentzVector currW = TLorentzVector(10000000.0, 10000000.0, 10000000.0, 10000000.0);
        for (int i = 0; i < FJets.size(); i++)
        {
            if (abs(wplusMass - FJets.at(i).M()) > abs(higgsMass - FJets.at(i).M()))
            {
                if(abs(higgsMass - FJets.at(i).M()) < abs(higgsMass - Higgs.M()))
                {
                    Higgs = FJets.at(i);
                    indexHiggs = i;
                }
            }
            else
            {
                if(abs(wplusMass - FJets.at(i).M()) < abs(wplusMass - W_hadronic.M()))
                {
                    W_hadronic = FJets.at(i);
                    indexWhadronic = i;
                }
            }
        }
        if (indexWhadronic != -1 && indexHiggs != -1)
	    {
	      //m_NTags_Higgs = nTaggedVRTrkJetsInFJet.at(indexHiggs);
	      //m_NTags_Wplus = nTaggedVRTrkJetsInFJet.at(indexWhadronic);

            // Apply small-R jet removal with respect to Higgs large-R jet
            ApplySmallRjetRemoval(&Jets, &JetIsTaggedBin, Higgs);
            // Apply small-R jet removal with respect to W boson large-R jet
            ApplySmallRjetRemoval(&Jets, &JetIsTaggedBin, W_hadronic);
          Xbb_variable_FJet_Higgs = log(ljet_Xbb2020v3_Higgs_REARRANGED.at(indexHiggs)/(0.25*ljet_Xbb2020v3_Top_REARRANGED.at(indexHiggs) + 0.75*ljet_Xbb2020v3_QCD_REARRANGED.at(indexHiggs)));
          Xbb_variable_FJet_WHad = log(ljet_Xbb2020v3_Higgs_REARRANGED.at(indexWhadronic)/(0.25*ljet_Xbb2020v3_Top_REARRANGED.at(indexWhadronic) + 0.75*ljet_Xbb2020v3_QCD_REARRANGED.at(indexWhadronic)));
	      
	      m_NTags = m_NTags_trkJ+m_NTags_caloJ;
	      m_ntagsOutside = m_NTags - (m_NTags_Higgs + m_NTags_Wplus);
	      m_bTagCategory = GetBTagCategory(m_NTags_Higgs, m_ntagsOutside);
          selection_category = 5;
          if (debugMode) std::cout << "\t\t\t\t" << "Set selection_category: " << selection_category << std::endl;
	    }
        if (indexWhadronic == -1)
	    {
            // Apply small-R jet removal with respect to Higgs large-R jet
            ApplySmallRjetRemoval(&Jets, &JetIsTaggedBin, Higgs);
	      W_hadronic = GetWFromJets();
	      //m_NTags_Higgs = nTaggedVRTrkJetsInFJet.at(indexHiggs);
          Xbb_variable_FJet_Higgs = log(ljet_Xbb2020v3_Higgs_REARRANGED.at(indexHiggs)/(0.25*ljet_Xbb2020v3_Top_REARRANGED.at(indexHiggs) + 0.75*ljet_Xbb2020v3_QCD_REARRANGED.at(indexHiggs)));
	      m_NTags = m_NTags_trkJ+m_NTags_caloJ;
	      m_ntagsOutside = m_NTags - (m_NTags_Higgs + N_bInW);
	      m_bTagCategory = GetBTagCategory(m_NTags_Higgs, m_ntagsOutside);
	      //	      std::cout<<"after"<<std::endl;//  return ;
          selection_category = 6;
          if (debugMode) std::cout << "\t\t\t\t" << "Set selection_category: " << selection_category << std::endl;
	    }
	    if (indexHiggs == -1)
	    {
            // Apply small-R jet removal with respect to W boson large-R jet
            ApplySmallRjetRemoval(&Jets, &JetIsTaggedBin, W_hadronic);
	      Higgs = GetHFromJets();
	      //m_NTags_Wplus = nTaggedVRTrkJetsInFJet.at(indexWhadronic);
          Xbb_variable_FJet_WHad = log(ljet_Xbb2020v3_Higgs_REARRANGED.at(indexWhadronic)/(0.25*ljet_Xbb2020v3_Top_REARRANGED.at(indexWhadronic) + 0.75*ljet_Xbb2020v3_QCD_REARRANGED.at(indexWhadronic)));
	      
	      m_NTags = m_NTags_trkJ+m_NTags_caloJ;
          m_ntagsOutside = m_NTags - (m_NTags_Wplus + N_bInH);
          m_bTagCategory = GetBTagCategory(m_NTags_Higgs, m_ntagsOutside);
          selection_category = 7;
          if (debugMode) std::cout << "\t\t\t\t" << "Set selection_category: " << selection_category << std::endl;
	    // return ;
	    
	    }

    }
    if (debugMode) std::cout << "\t\t\t" << "m_NTags = " << m_NTags << std::endl;
    if (debugMode) std::cout << "\t\t\t" << "m_NTags_trkJ = " << m_NTags_trkJ << std::endl;
    if (debugMode) std::cout << "\t\t\t" << "m_NTags_Higgs = " << m_NTags_Higgs << std::endl;
    if (debugMode) std::cout << "\t\t\t" << "m_ntagsOutside = " << m_ntagsOutside << std::endl;
    if (debugMode) std::cout << "\t\t\t" << "m_bTagCategory = " << m_bTagCategory << std::endl;
    if (debugMode) std::cout << "\t\t" << "Leaving SetJetPair" << std::endl;
}


void EventLoop::ApplySmallRjetRemoval(std::vector<TLorentzVector> *smallRJets, std::vector<int> *smallRJetsIsTagged, TLorentzVector largeRJet){
    /*
        Remove elements from both smallRJets, smallRJetsTagged (these should be of same length) according to if the
        smallRjet is too close to the largeRjet or not
    */
    if (debugMode) std::cout << "\t\t" << "Entering ApplySmallRjetRemoval" << std::endl;
    if (debugMode) std::cout << "\t\t\t" << "Beginning loop over " << smallRJets->size() << " jets" << std::endl;
    
    std::vector<int> removal;
    if (debugMode) std::cout << "\t\t\t" << "Comparing to large-R jet with Pt=" << largeRJet.Pt() << " and m=" << largeRJet.M() << std::endl;
    for (unsigned int i = 0; i < smallRJets->size(); i++) {
        if (debugMode) std::cout << "\t\t\t" << "Considering small-R jet #" << i << "with Pt=" << smallRJets->at(i).Pt() << " and m=" << smallRJets->at(i).M() << std::endl;
        if (largeRJet.DeltaR(smallRJets->at(i)) < Small_Jet_removal){
            if (debugMode) std::cout << "\t\t\t" << "Removing small-R jet since deltaR with large-R jet is " << largeRJet.DeltaR(smallRJets->at(i)) << " < " << Small_Jet_removal << std::endl;
            removal.push_back(i);
        }
    }
    for (unsigned int i=0; i<removal.size(); i++){
        if (debugMode) std::cout << "\t\t\t" << "ACTUALLY Removing small-R jet #" << i << "with Pt=" << smallRJets->at(removal[i]-i).Pt() << " and m=" << smallRJets->at(removal[i]-i).M() << std::endl;
        smallRJets->erase(smallRJets->begin()+removal[i]-i);
        smallRJetsIsTagged->erase(smallRJetsIsTagged->begin()+removal[i]-i);
    }

    if (debugMode) std::cout << "\t\t\t" << "Finished loop over small-R jets" << std::endl;
    if (debugMode) std::cout << "\t\t" << "Leaving ApplySmallRjetRemoval" << std::endl;
}




void EventLoop::FindTop()
{
    if (debugMode) std::cout << "\t\t" << "Entering FindTop" << std::endl;
    const double_t topMass = 172760;

    // Loop through jets, chekc if tagged as B and if so test the mass for being close to m_top. Choose closest
    // TODO is it also worth doing a delta R check or something there?
    // First, do so for hadronic W
    double Diff=100000000;
    Top_hadro.SetPtEtaPhiM(999999,999999,999999,999999);
    int hadro_top_b_index = -1;
    if (debugMode) std::cout << "\t\t" << "Beginning Hadro loop over " << Jets.size() << " jets" << std::endl;
    for (unsigned int j = 0; j < Jets.size(); j++){
        if (debugMode) std::cout << "\t\t" << "Hadro loop, jet " << j << std::endl;
        //if (JetIsTaggedContinuous.at(j) > m_btagCut_value_CaloJets)
        if (JetIsTaggedBin.at(j) >= m_btagCategoryBin){
            if (debugMode) std::cout << "\t\t" << "Hadro loop, jet " << j << " in bin " << JetIsTaggedBin.at(j) << " so passed btag thresh: " << m_btagCategoryBin << std::endl;
            Top_hadro = W_hadronic + Jets.at(j);
            if (debugMode) std::cout << "\t\t" << "Top_hadro mass candidate: " << Top_hadro.M() << std::endl;
            if (abs(topMass-Top_hadro.M())<Diff){
                if (debugMode) std::cout << "\t\t" << "Diff decreased from" << Diff << " to " << abs(topMass-Top_hadro.M()) << ", updating Whadro to use bjet " << j << std::endl;
                hadro_top_b_index=j;
                Diff=abs(topMass-Top_hadro.M());
            }
        }
    }
    if (hadro_top_b_index >= 0){
        Top_hadro = W_hadronic + Jets.at(hadro_top_b_index);
        m_mTop_hadro = Top_hadro.M();
        if (debugMode) std::cout << "\t\t\t" << "Top_hadro candidate found, mass = " << Top_hadro.M() << std::endl;
    }
    else{
        if (debugMode) std::cout << "\t\t\t" << "No good candidates found for b-quark for hadronic top, setting to bad Lorentz Vector" << std::endl;
        //m_mTop_hadro = -1; Defaults to -1
        Top_hadro.SetPtEtaPhiM(0,0,0,0);
    }
    
    Diff=100000000;
    double diff_t;
    if (debugMode) std::cout << std::endl;
    // Now same for leptonic W
    Top_lepto.SetPtEtaPhiM(0,0,0,0);
    int lepto_top_b_index = -1;
    if (debugMode) std::cout << "\t\t" << "Beginning Lepto loop over " << Jets.size() << " jets" << std::endl;
    for (unsigned int j = 0; j < Jets.size(); j++){
        if (debugMode) std::cout << "\t\t" << "Lepto loop, jet " << j << std::endl;
        //if (JetIsTaggedContinuous.at(j) > m_btagCut_value_CaloJets)
        //if (JetIsTaggedBin.at(j) >= m_btagCategoryBin){
        if (true) {
            if (debugMode) std::cout << "\t\t" << "Lepto loop, jet " << j << " in bin " << JetIsTaggedBin.at(j) << " so passed btag thresh: " << m_btagCategoryBin << std::endl;
            Top_lepto = W_leptonic + Jets.at(j);
            if (debugMode) std::cout << "\t\t" << "Top_lepto mass candidate: " << Top_lepto.M() << std::endl;
            diff_t = abs(topMass-Top_lepto.M())/(0.15*Top_lepto.M());
            if ((diff_t<Diff) & Top_lepto.M() > 0){
                if (debugMode) std::cout << "\t\t" << "Diff decreased from" << Diff << " to " << diff_t << ", updating Wlepto to use bjet " << j << std::endl;
                if (debugMode) std::cout << "This was done using jet #" << j << " with Pt=" << Jets.at(j).Pt() << " and m=" << Jets.at(j).M() << std::endl;
                lepto_top_b_index=j;
                Diff=diff_t;
            }
        }
    }
    if (lepto_top_b_index >= 0){
        Top_lepto = W_leptonic + Jets.at(lepto_top_b_index);
        m_mTop_lepto = Top_lepto.M();
        if (debugMode) std::cout << "\t\t\t" << "Top_lepto candidate found, mass = " << Top_lepto.M() << std::endl;
    }
    else{
        Top_lepto.SetPtEtaPhiM(999999,999999,999999,999999);
        //m_mTop_lepto = -1; Defaults to -1
        if (debugMode) std::cout << "\t\t\t" << "No good candidates found for b-quark for leptonic top, setting to bad Lorentz Vector" << std::endl;
    }
    if (debugMode) std::cout << "\t\t\t" << "Truth W decay mode: " << truth_W_decay_mode << std::endl;

    m_mTop_best = (abs(topMass-Top_hadro.M()) > abs(topMass-Top_lepto.M())) ? Top_lepto.M() : Top_hadro.M();


    if (debugMode) std::cout << "\t\t" << "Leaving FindTop" << std::endl;
}


TLorentzVector EventLoop::GetWFromJets()
{
  if (debugMode) std::cout << "\t\t" << "Entering GetWFromJets" << std::endl;
  if (Jets.size() < 2){
    if (debugMode) std::cout << "\t\t\t" << "Number of small-R jets remaining (after all large-R jet overlap removals) = " << Jets.size() << " < 2, so setting the W to be nonsense TLorentzVector(1000,1000,1000,1000)" << std::endl;
    TLorentzVector W = TLorentzVector(1000,1000,1000,1000);
    return W;
  }
  //assert(("Must be at least 2 jets to run GetWFromJets function", Jets.size() >=2));
  //std::cout<<"in Wfromjet"<<std::endl;
  const double_t wplusMass = 80379;  
  double Diff=100000000;
  index_W_jet_1=-1;
  index_W_jet_2=-1;
  N_bInW = 0;
  TLorentzVector W;
  for (unsigned int i = 0; i < Jets.size(); i++) {
      for (unsigned int j = i + 1; j < Jets.size(); j++) {
	  W = Jets.at(i) + Jets.at(j);
	  if (abs(wplusMass-W.M())<Diff) {
	      index_W_jet_1=i;
	      index_W_jet_2=j;
	      Diff=abs(wplusMass-W.M());
	    }
      }
    }

  //if (jet_DL1r->at(index_W_jet_1) > m_btagCut_value_CaloJets)
  if (JetIsTaggedBin.at(index_W_jet_1) >= m_btagCategoryBin) {
      N_bInW ++;
    }
  //if (jet_DL1r->at(index_W_jet_2) > m_btagCut_value_CaloJets)
  if (JetIsTaggedBin.at(index_W_jet_2) >= m_btagCategoryBin) {
      N_bInW ++;
    }

  W = Jets.at(index_W_jet_1) + Jets.at(index_W_jet_2);
  if (debugMode) std::cout << "\t\t" << "Leaving GetWFromJets" << std::endl;
  return W;
}


TLorentzVector EventLoop::GetHFromJets()
{
  if (debugMode) std::cout << "\t\t" << "Entering GetHFromJets" << std::endl;
  if (Jets.size() < 2){
    if (debugMode) std::cout << "\t\t\t" << "Number of small-R jets remaining (after all large-R jet overlap removals) = " << Jets.size() << " < 2, so setting the Higgs to be nonsense TLorentzVector(1000,1000,1000,1000)" << std::endl;
    TLorentzVector H = TLorentzVector(1000,1000,1000,1000);
    return H;
  }
  //assert(("Must be at least 2 jets to run GetHFromJets function", Jets.size() >=2));
  const double_t higgsMass = 125100;
  N_bInH = 0;
  double Diff=100000000;
  index_H_jet_1=-1;
  index_H_jet_2=-1;
  TLorentzVector H;
  for (unsigned int i = 0; i < Jets.size(); i++)
    {
      for (unsigned int j = i + 1; j < Jets.size(); j++)
        {
	  H = Jets.at(i) + Jets.at(j);
	  if (abs(higgsMass-H.M())<Diff)
	    {
	      index_H_jet_1=i;
	      index_H_jet_2=j;
	      Diff=abs(higgsMass-H.M());
	    }
	}
    }

  //if (jet_DL1r->at(index_H_jet_1) > m_btagCut_value_CaloJets)
  if (JetIsTaggedBin.at(index_H_jet_1) >= m_btagCategoryBin)
    {
      N_bInH ++;
    }
  //if (jet_DL1r->at(index_H_jet_2) > m_btagCut_value_CaloJets)
  if (JetIsTaggedBin.at(index_H_jet_2) >= m_btagCategoryBin)
    {
      N_bInH ++;
    }

  H = Jets.at(index_H_jet_1) + Jets.at(index_H_jet_2);
  if (debugMode) std::cout << "\t\t" << "Leaving GetHFromJets" << std::endl;
  return H;
}


TLorentzVector EventLoop::GetWBoson(bool &status)
{
    if (debugMode) std::cout << "\t" << "Entering GetWBoson" << std::endl;
    status = false;
    TLorentzVector Wleptonic;
   //if(Leptons.size() == 0) return Wleptonic; //TODO is this a sensible check? Simon does it
    std::vector<TLorentzVector *> neutrinoVector = GetNeutrinos(&Leptons.at(0), &MET);
    for (auto neutrino : neutrinoVector) // TODO why loop over all neutrinos here?
    {
        Wleptonic = (*neutrino + Leptons.at(0));
        status = true;
        if (debugMode) std::cout << "\t\t" << "Got W Boson with Pt = " << Wleptonic.Pt() << std::endl;
    }
    if (debugMode) std::cout << "\t" << "Leaving GetWBoson" << std::endl;
    return Wleptonic;
}

std::vector<TLorentzVector *> EventLoop::GetNeutrinos(TLorentzVector *L, TLorentzVector *MET)
{
    if (debugMode) std::cout << "\t\t" << "Entering GetNeutrinos" << std::endl;
    std::vector<TLorentzVector *> neutrinoVector;
    neutrinoVector = m_NeutrinoBuilder->candidatesFromWMass_Rotation(L, MET, true);
    bool m_isRotatedSol = m_NeutrinoBuilder->m_isRotated;
    double m_r = m_NeutrinoBuilder->m_r;
    if (debugMode) std::cout << "\t\t" << "Leaving GetNeutrinos" << std::endl;
    return neutrinoVector;
}


void EventLoop::Set_Jet_observables()
{
    m_min_DeltaPhiJETMET = 999;
    m_HT = 0;
    m_HT_bjets = 0;
    m_maxEta_bjets = 0;
    m_maxPT_bjets = 0;
    int i = 0;
    for (auto jet : Jets)
    {
        double DeltaPhiJETMET = fabs(jet.DeltaPhi(MET));
        if (DeltaPhiJETMET < m_min_DeltaPhiJETMET)
        {
            m_min_DeltaPhiJETMET = DeltaPhiJETMET;
        }
        m_HT += jet.Pt() * 0.001;
        //if (JetIsTaggedContinuous.at(i) >= m_btagCut_value_CaloJets)
        if (JetIsTaggedBin.at(i) >= m_btagCategoryBin)
        {
            m_HT_bjets += jet.Pt() * 0.001;
            if (m_maxEta_bjets < fabs(jet.Eta()))
            {
                m_maxEta_bjets = fabs(jet.Eta());
            }
            if (m_maxPT_bjets < jet.Pt() * 0.001)
            {
                m_maxPT_bjets = jet.Pt() * 0.001;
            }
        }
        i++;
    }
}

void EventLoop::CutFlowAssignment(CutFlowType &cutVariable, bool XUnweightedCutFlow, bool XWeightedCutFlow)
{
    if (XUnweightedCutFlow)
    {
        cutVariable.increment_unweighted("both_channels", m_NTags);
        if (is_signal_sample && truth_W_decay_mode == "qqbb") cutVariable.increment_unweighted("jjbb_channel", m_NTags);
        if (is_signal_sample && truth_W_decay_mode == "lvbb") cutVariable.increment_unweighted("lvbb_channel", m_NTags);
    }
    if (XWeightedCutFlow)
    {
        cutVariable.increment_weighted("both_channels", m_NTags, EventWeight);
        if (is_signal_sample && truth_W_decay_mode == "qqbb") cutVariable.increment_weighted("jjbb_channel", m_NTags, EventWeight);
        if (is_signal_sample && truth_W_decay_mode == "lvbb") cutVariable.increment_weighted("lvbb_channel", m_NTags, EventWeight);
    }
}

void EventLoop::altCutFlow(Float_t met_pt_min, Float_t lep_pt_min, Float_t higgs_pt_min, Float_t W_leptonic_pt_min, Float_t lep_SMHiggs_angle_min, Float_t lep_SMHiggs_angle_max, Float_t lep_W_hadronic_angle_min,
                           Float_t hw_angle)
{
    return; //This is a place for testing alternate event selection methods
}

bool EventLoop::PassEventSelectionBoosted(Float_t met_pt_min, Float_t lep_pt_min, Float_t higgs_pt_min, Float_t W_leptonic_pt_min, Float_t lep_SMHiggs_angle_min, Float_t lep_SMHiggs_angle_max, Float_t lep_W_hadronic_angle_min,
                                          Float_t hw_angle)
{
    if (debugMode) std::cout << "\t" << "Entering PassEventSelectionBoosted function" << std::endl;
    //if (Lepton_Charge < 0)

    bool tmp_bool = false;
    W_leptonic.SetPtEtaPhiM(99999999,99999999,99999999,99999999);
    W_leptonic = GetWBoson(tmp_bool);
    if ((!tmp_bool)  && status_W_cut_on)
    {
        if (debugMode) std::cout << "\t\t\t" << "Cutflow: PositiveLepWCutflow failed, discarding event. status_W = " << tmp_bool << std::endl;
        CutFlowAssignment(m_PositiveLepWCutflow, UnweightedCutFlow, WeightedCutFlow);
        return false;
    }

    if (debugMode) std::cout << "\t\t" << "Set W_leptonic: " << tmp_bool << std::endl;
    SetJetPair();
    // Commenting this out for now; no longer have requirement on NTags
    /*
    if (m_NTags == 0 || m_NTags == 1 || m_NTags == -1)
    {
        if (debugMode) std::cout << "\t\t" << "Failed - insufficient ntags found; number of nTags = " << m_NTags << std::endl;
        return false;
    }
    if (debugMode) std::cout << "\t\t" << "Passed - sufficient ntags found" << std::endl;
    */
    CutFlowAssignment(m_TotalEvents, UnweightedCutFlow, WeightedCutFlow); // This actually counts the
    if (FJets.size() < min_n_fat_jets && min_n_fat_jets_cut_on)
    {
        CutFlowAssignment(m_MinNFatJetsCutFlow, UnweightedCutFlow, WeightedCutFlow);
        if (debugMode) std::cout << "\t\t" << "Cutflow: MinNFatJetsCutFlow failed, discarding event. #Fatjets = " << FJets.size() << " < " << min_n_fat_jets << std::endl;
        return false;
    }
    if (FJets.size() > max_n_fat_jets && max_n_fat_jets_cut_on)
    {
        CutFlowAssignment(m_MaxNFatJetsCutFlow, UnweightedCutFlow, WeightedCutFlow);
        if (debugMode) std::cout << "\t\t" << "Cutflow: MaxNFatJetsCutFlow failed, discarding event. #Fatjets = " << FJets.size() << " > " << max_n_fat_jets << std::endl;
        return false;
    }
    altCutFlow(met_pt_min, lep_pt_min, higgs_pt_min, W_leptonic_pt_min, lep_SMHiggs_angle_min, lep_SMHiggs_angle_max, lep_W_hadronic_angle_min,
               hw_angle);
    if (MET.Pt() < met_pt_min && met_pt_min_cut_on)
    {
        CutFlowAssignment(m_METCutFlow, UnweightedCutFlow, WeightedCutFlow);
        if (debugMode) std::cout << "\t\t" << "Cutflow: METCutFlow failed, discarding event. MET Pt = " << MET.Pt() << std::endl;
        // selection_category = -2;
        return false;
    }
    if (Leptons.at(0).Pt() < lep_pt_min && lep_pt_min_cut_on)
    {
        CutFlowAssignment(m_LeptonPtCutFlow, UnweightedCutFlow, WeightedCutFlow);
        if (debugMode) std::cout << "\t\t" << "Cutflow: LeptonPtCutFlow failed, discarding event. Lepton Pt = " << Leptons.at(0).Pt() << std::endl;
        // selection_category = -3;
        return false;
    }
    if (W_leptonic.Pt() < W_leptonic_pt_min && W_leptonic_pt_min_cut_on)
    {
        CutFlowAssignment(m_WPtCutFlow, UnweightedCutFlow, WeightedCutFlow);
        if (debugMode) std::cout << "\t\t" << "Cutflow: W_leptonic_PtCutFlow failed, discarding event. W_lep Pt = " << W_leptonic.Pt() << std::endl;
        // selection_category = -4;
        return false;
    }
    if (category_cut_on) {
        if (std::find(chosen_categories.begin(), chosen_categories.end(), selection_category) == chosen_categories.end())
        {
            CutFlowAssignment(m_selectionCategoryCutFlow, UnweightedCutFlow, WeightedCutFlow);
            if (debugMode) std::cout << "\t\t" << "Cutflow: selection_categoryCutFlow failed, discarding event. selection_category = " << selection_category << std::endl;
            return false;
        }
        //if (debugMode) std::cout << "\t\t" << "Cutflow: selection_categoryCutFlow passed, keeping event. selection_category = " << selection_category << std::endl;
    }

    if (debugMode) std::cout << "\t" << "Leaving PassEventSelectionBoosted function (via FindFJetPair)" << std::endl;
    bool status_FindFJetPair;
    status_FindFJetPair = FindFJetPair(higgs_pt_min, W_leptonic_pt_min, lep_SMHiggs_angle_min, lep_SMHiggs_angle_max, lep_W_hadronic_angle_min, hw_angle);
    if (!status_FindFJetPair) return false;
    return true;
}


double EventLoop::GetMwt() // Is Mwt Transverse mass of the W? TODO Check this. Answer: Yes, see https://arxiv.org/pdf/1701.07240.pdf page 4
{
    return 0.001 * sqrt(2. * Leptons.at(0).Pt() * MET.Pt() * (1. - cos(Leptons.at(0).DeltaPhi(MET))));
}

bool EventLoop::SetLeptonVectors(){ // What if there is more than one Lepton?
    if (debugMode) std::cout << "\t" << "Entering SetLeptonVectors" << std::endl;
   Leptons.clear();
   TLorentzVector lepton;
   bool trigger_matched = true;
   if (debugMode) std::cout << "\t\t" << "Looping over " << mu_pt->size() << " muons" << std::endl;
   for(int i=0; i < mu_pt->size(); i++) {
        if(mu_pt->at(i)*0.001 < 27.)continue;
        ///if(((int) mu_isoFCTightTrackOnly->at(i)) != 1)continue;
        //if(((int) mu_isTight->at(i)) != 1)continue; Updated to the below line
        if(mu_pt->at(i)*0.001 < 300 && ((int) mu_isTight->at(i)) != 1)continue;
        if(fabs(mu_d0sig->at(i)) > 3.0)continue;
        lepton.SetPtEtaPhiE(mu_pt->at(i), mu_eta->at(i), mu_phi->at(i), mu_e->at(i));
        Leptons.push_back(lepton);
        ///trigger_matched = ((int) mu_trigMatch_HLT_mu26_ivarmedium->at(i) || (int) mu_trigMatch_HLT_mu50->at(i));
   }
   if (debugMode) std::cout << "\t\t" << "Looping over " << el_pt->size() << " electrons" << std::endl;
   for(int i=0; i < el_pt->size(); i++) {
        if(el_pt->at(i)*0.001 < 27.)continue;
        if(((int) el_LHTight->at(i)) != 1)continue;
        ///if(((int) el_isoFCTight->at(i)) != 1)continue;
        if(fabs(el_d0sig->at(i)) > 5.)continue;
        ///if(fabs(el_delta_z0_sintheta) > 0.5)continue;
        lepton.SetPtEtaPhiE(el_pt->at(i), el_eta->at(i), el_phi->at(i), el_e->at(i));
        Leptons.push_back(lepton);
        ///trigger_matched	= ((int) el_trigMatch_HLT_e26_lhtight_nod0_ivarloose->at(i) || (int) el_trigMatch_HLT_e60_lhmedium_nod0->at(i) || (int) el_trigMatch_HLT_e140_lhloose_nod0->at(i));
   }
   if (debugMode) std::cout << "\t\t" << "Found " << Leptons.size() << " leptons" << std::endl;
   if (Leptons.size() != 1) return false;
   // Check that it's triggered - NOTE We removed this because it was removing all 2015 events; see https://indico.cern.ch/event/1284084/contributions/5395442/attachments/2659343/4606255/Boosted_Update_05_06_2023.pdf
//    if (!(HLT_mu26_ivarmedium || HLT_mu50 || HLT_e60_lhmedium_nod0 || HLT_e140_lhloose_nod0 || HLT_e26_lhtight_nod0_ivarloose)){
//     if (debugMode) std::cout << "\t\t" << "Failed is_Triggered, so returning false" << std::endl;
//     return false;
//    }
   Lepton_Eta = Leptons.at(0).Eta();
   Lepton_Pt = Leptons.at(0).Pt() * 0.001;
   Lepton_Phi = Leptons.at(0).Phi();
   Lepton_M = Leptons.at(0).M();
    if (debugMode) std::cout << "\t" << "Leaving SetLeptonVectors" << std::endl;
   return true;
}

void EventLoop::SetJetVectors()
{
    /*
    Function to fill jet vectors with the info from the input ntuple.
    This function also fills some information about the number of tags.
    In this function, we refer to several different types of jets. These follow the naming
    convention from VH semileptonic:
    - 'Signal jets':        calorimeter jets with |eta| < 2.5
    - 'Forward jets':       calorimeter jets with |eta| > 2.5
    - 'Fat jets':           calorimeter jets with radius above a certain size
    - 'Track jets':         track jets (similar to calorimeter jets, but clustered with tracks instead of calo clusters)
    */
    if (debugMode) std::cout << "\t" << "Entering SetJetVectors" << std::endl;
    float DxbbThreshold = 2.44;
    float DXbb;
    bool small_jet_okay;
    Jets.clear();
    Jets_Pt.clear();
    Jets_Eta.clear();
    Jets_Phi.clear();
    Jets_M.clear();
    Jets_tagWeightBinDL1rContinuous.clear();
    //JetIsTagged.clear();
    JetIsTaggedBin.clear();
    TrkJets.clear();
    TrkJetIsTaggedBin.clear();
    FJets.clear();
    FJets_Pt.clear();
    FJets_Eta.clear();
    FJets_Phi.clear();
    FJets_M.clear();
    FJets_DXbb.clear();
    ljet_Xbb2020v3_Higgs_REARRANGED.clear();
    ljet_Xbb2020v3_QCD_REARRANGED.clear();
    ljet_Xbb2020v3_Top_REARRANGED.clear();
    //nTaggedVRTrkJetsInFJet.clear();
    TLorentzVector jet, trkjet, fjet;
    if (debugMode) std::cout << "\t\t" << "Beginning loop over " << signal_Jet_PT->size() << " signal jets" << std::endl;
    for (int i = 0; i < signal_Jet_PT->size(); i++) // Loop over the 'signal jets', which are calorimeter jets with |eta| < 2.5
    {
        //jet.SetPtEtaPhiM(signal_Jet_PT->at(i), signal_Jet_Eta->at(i), signal_Jet_Phi->at(i), signal_Jet_M->at(i));
        jet.SetPtEtaPhiE(signal_Jet_PT->at(i), signal_Jet_Eta->at(i), signal_Jet_Phi->at(i), signal_Jet_E->at(i)); // TODO why are we using setPtEtaPhiE instead of setPtEtaPhiM ?
        if (debugMode) std::cout << "\t\t\t" << "Pushing back signal_Jet #" << (i+1) << std::endl;
        Jets.push_back(jet);
        JetIsTaggedBin.push_back(signal_Jet_tagWeightBin_DL1r_Continuous->at(i));
        if (debugMode) std::cout << "\t\t\t" << "signal_Jet tagWeightBin_DL1r_Continuous = " << signal_Jet_tagWeightBin_DL1r_Continuous->at(i) << std::endl;
        if(signal_Jet_tagWeightBin_DL1r_Continuous->at(i) >= m_btagCategoryBin)
        {
            m_NTags_caloJ++;
            if (debugMode) std::cout << "\t\t\t" << "signal_Jet #" << (i+1) << " passed tagging threshold, adding to m_NTags_caloJ" << std::endl;
        }

        /*Commenting this out to move the small jet removal elsewhere, so that it only removes those close to W or Higgs candidate large-R jets
        small_jet_okay = true;
        for (int j = 0; j < FatJet_M->size(); j++)
        {
            fjet.SetPtEtaPhiM(FatJet_PT->at(j), FatJet_Eta->at(j), FatJet_Phi->at(j), FatJet_M->at(j));
            if (fjet.DeltaR(jet) < Small_Jet_removal)
            {
                small_jet_okay = false;
            }
        }
        if (small_jet_okay){
            if (lowlevel_output_mode){
                Jets_Pt.push_back(jet.Pt());
                Jets_Eta.push_back(jet.Eta());
                Jets_Phi.push_back(jet.Phi());
                Jets_M.push_back(jet.M());
                Jets_tagWeightBinDL1rContinuous.push_back(signal_Jet_tagWeightBin_DL1r_Continuous->at(i));
            }
        }
        */
    }
    if (debugMode) std::cout << "\t\t" << "Finished loop over signal jets" << std::endl;
    if (lowlevel_output_mode){
        Sort_vec1_by_vec2(&Jets_Eta, &Jets_Pt);
        Sort_vec1_by_vec2(&Jets_Phi, &Jets_Pt);
        Sort_vec1_by_vec2(&Jets_M, &Jets_Pt);
        Sort_vec1_by_vec2(&Jets_tagWeightBinDL1rContinuous, &Jets_Pt);
        Sort_vec1_by_vec2(&Jets_Pt, &Jets_Pt);
    }
    /* // In the boosted channel, we don't seem to use the forward jets
    for (int i = 0; i < forward_Jet_PT->size(); i++) // Loop over the 'forward jets', which are calorimeter jets with |eta| > 2.5
    {
        jet.SetPtEtaPhiM(forward_Jet_PT->at(i), forward_Jet_Eta->at(i), forward_Jet_Phi->at(i), forward_Jet_M->at(i));
        Jets.push_back(jet);
        JetIsTagged.push_back(GetTagWeightBin(btag_score_forwardJet->at(i)));
        if (btag_score_forwardJet->at(i) > m_btagCut_value_CaloJets)
            m_NTags_caloJ++;
    }
    */
    if (debugMode) std::cout << "\t\t" << "Beginning loop over " << FatJet_M->size() << " fat jets" << std::endl;
    for (int i = 0; i < FatJet_M->size(); i++) // Loop over the 'fat jets', which are calorimeter jets with radius above a certain size
    {
        if (FatJet_PT->at(i) < 250000)
        {
            if (debugMode) std::cout << "\t\t\t" << "FatJet #" << (i+1) << " discarded, Pt = " << FatJet_PT->at(i) << " < 250000" << std::endl;
            continue;
        }
        if (FatJet_M->at(i) < 50000)
        {
            if (debugMode) std::cout << "\t\t\t" << "FatJet #" << (i+1) << " discarded, M = " << FatJet_M->at(i) << " < 50000" << std::endl;
            continue;
        }
        if (FatJet_M->at(i) > 250000)
        {
            if (debugMode) std::cout << "\t\t\t" << "FatJet #" << (i+1) << " discarded, M = " << FatJet_M->at(i) << " > 250000" << std::endl;
            continue;
        }
        if (debugMode) std::cout << "\t\t\t" << "FatJet #" << (i+1) << " passed, pushing back" << std::endl;
        fjet.SetPtEtaPhiM(FatJet_PT->at(i), FatJet_Eta->at(i), FatJet_Phi->at(i), FatJet_M->at(i));
        if (fjet.DeltaR(Leptons.at(0)) < 1.0)
        {
            if (debugMode) std::cout << "\t\t\t" << "FatJet #" << (i+1) << " discarded, deltaR with Lepton = " << fjet.DeltaR(Leptons.at(0)) << " < 1.0" << std::endl;
            continue;
        }
        FJets.push_back(fjet);
        ljet_Xbb2020v3_Higgs_REARRANGED.push_back(ljet_Xbb2020v3_Higgs->at(i));
        ljet_Xbb2020v3_QCD_REARRANGED.push_back(ljet_Xbb2020v3_QCD->at(i));
        ljet_Xbb2020v3_Top_REARRANGED.push_back(ljet_Xbb2020v3_Top->at(i));

        // Push to low-level-input branches of the tree
        if (lowlevel_output_mode){
            DXbb = log(ljet_Xbb2020v3_Higgs->at(i)/(0.25*ljet_Xbb2020v3_Top->at(i) + 0.75*ljet_Xbb2020v3_QCD->at(i)));
            FJets_Pt.push_back(fjet.Pt());
            FJets_Eta.push_back(fjet.Eta());
            FJets_Phi.push_back(fjet.Phi());
            FJets_M.push_back(fjet.M());
            FJets_DXbb.push_back(DXbb);
        }
        /*
        int nMatchedAndTaggedJets = 0;
        for (int j = 0; j < TrackJet_PT->size(); j++) // Loop over the 'track jets', which are track jets (similar to calorimeter jets, but clustered with tracks instead of calo clusters)
        {
            //trkjet.SetPtEtaPhiM(TrackJet_PT->at(i), TrackJet_Eta->at(i), TrackJet_Phi->at(i), TrackJet_M->at(i));
            trkjet.SetPtEtaPhiE(TrackJet_PT->at(j), TrackJet_Eta->at(j), TrackJet_Phi->at(j), TrackJet_E->at(j));
            //if (debugMode) std::cout << "track_Jet_tagWeightBin_DL1r_Continuous: " << track_Jet_tagWeightBin_DL1r_Continuous->at(i) << std::endl; 
            if (fjet.DeltaR(trkjet) < 1.0 && track_Jet_tagWeightBin_DL1r_Continuous->at(j) >= m_btagCategoryBin)
            {
                if (debugMode) std::cout << "\t\t\t" << "TrackJet #" << (j+1) << " matched to FatJet #" << (i+1) << std::endl;
                nMatchedAndTaggedJets++;
            }
        }
        nTaggedVRTrkJetsInFJet.push_back(nMatchedAndTaggedJets);
        */
    }
    if (debugMode) std::cout << "\t\t" << "Finished loop over large-R jets" << std::endl;
    if (lowlevel_output_mode){
        Sort_vec1_by_vec2(&FJets_Eta, &FJets_Pt);
        Sort_vec1_by_vec2(&FJets_Phi, &FJets_Pt);
        Sort_vec1_by_vec2(&FJets_M, &FJets_Pt);
        Sort_vec1_by_vec2(&FJets_DXbb, &FJets_Pt);
        Sort_vec1_by_vec2(&FJets_Pt, &FJets_Pt);
    }

    if (debugMode) std::cout << "\t\t" << "Beginning loop over " << TrackJet_PT->size() << " track jets" << std::endl;
    for (int i = 0; i < TrackJet_PT->size(); i++)
    {
        //jet.SetPtEtaPhiM(TrackJet_PT->at(i), TrackJet_Eta->at(i), TrackJet_Phi->at(i), TrackJet_M->at(i));
        jet.SetPtEtaPhiE(TrackJet_PT->at(i), TrackJet_Eta->at(i), TrackJet_Phi->at(i), TrackJet_E->at(i));
        TrkJets.push_back(jet);
        TrkJetIsTaggedBin.push_back(track_Jet_tagWeightBin_DL1r_Continuous->at(i));
        if (debugMode) std::cout << "\t\t\t" << "TrackJet #" << (i+1) << " has tagWeightBin_DL1r_Continuous = " << track_Jet_tagWeightBin_DL1r_Continuous->at(i) << std::endl;
        if (track_Jet_tagWeightBin_DL1r_Continuous->at(i) >= m_btagCategoryBin)
        {
            //TrkJetIsTaggedBin.push_back(1); // Commenting out since I think this is unused (TODO confirm)
            m_NTags_trkJ++;
            if (debugMode) std::cout << "\t\t\t" << "TrackJet #" << (i+1) << " passed btagCategoryBin_trkJets criteria, adding to NTags_trkJ" << std::endl;
        }
        else
        {
            //TrkJetIsTaggedBin.push_back(0); // Commenting out since I think this is unused (TODO confirm)
            if (debugMode) std::cout << "\t\t\t" << "TrackJet #" << (i+1) << " failed btagCategoryBin_trkJets criteria" << std::endl;
        }
    }
    if (debugMode) std::cout << "\t\t" << "Finished loop over track jets" << std::endl;
    //Sort_Jets_and_vars(&FJets, &nTaggedVRTrkJetsInFJet, &ljet_Xbb2020v3_Higgs_REARRANGED, &ljet_Xbb2020v3_QCD_REARRANGED, &ljet_Xbb2020v3_Top_REARRANGED); // TODO do we even need to sort these??
    //Sort_Jets(&Jets, &JetIsTaggedBin); // TODO do we even need to sort these??
    if (debugMode) std::cout << "\t" << "Leaving SetJetVectors" << std::endl;
}


void EventLoop::btagCounting(){
    if (debugMode) std::cout << "\t\t" << "Entering btagCounting" << std::endl;
    m_nTrkTagsOutside = 0;
    m_nTrkTagsInW = 0;
    m_nTrkTagsInH = 0;
    //assert(("Higgs not yet reconstructed but trying to do btag counting", higgs_reconstructed)) // Example of a sensible assertion to ahve to make sure we don't do anything stupid
    // Our strategy here depends on the selection category, since this will determine if the Higgs/W has been constructed from Fat jets or not
    // Remember categories:
    //      0) Higgs large-R jet,  Whadro small-R jets
    //      1) Higgs large-R jet,  Whadro small-R jets
    //      2) Higgs small-R jets, Whardo large-R jet
    //      3) Higgs large-R jet,  Whadro large-R jet
    //      4) Higgs large-R jet,  Whadro large-R jet
    //      5) Higgs large-R jet,  Whadro large-R jet
    //      6) Higgs large-R jet,  Whadro small-R jets
    //      7) Higgs small-R jets, Whadro large-R jet

    /*
    // NB   Have to use calo jets, since these are what were used to define the W/Higgs if they were defined using small jets, so to match up need to do that
    //      If, however, we decide not to use the information about which jets were used to reconstruct, and instead to re-check which jets are closest, then can use
    //          track jets. However, this allows the possibility of a jet which was used to construct e.g. the W to be counted as a btag inside the Higgs
    if (debugMode) std::cout << "\t\t" << "Beginning loop over " << Jets.size() << " Calo jets" << std::endl;
        for (int i = 0; i < Jets.size(); i++) {
            if (debugMode) std::cout << "\t\t\t" << "CaloJet #" << (i+1) << " has tagWeightBin_DL1r_Continuous = " << JetIsTaggedBin.at(i) << std::endl;
            if (i == index_H_jet_1 || i == index_H_jet_2 || i == index_W_jet_1 || i == index_W_jet_2) continue; // These have already been accounted for
            if (JetIsTaggedBin.at(i) >= m_btagCategoryBin)
            {
                if Higgs.Small_Jet_removal
            }
            else
            {
                //TrkJetIsTaggedBin.push_back(0); // Commenting out since I think this is unused (TODO confirm)
                if (debugMode) std::cout << "\t\t\t" << "TrackJet #" << (i+1) << " failed btagCategoryBin_trkJets criteria" << std::endl;
            }
    }
    if (debugMode) std::cout << "\t\t" << "Finished loop over track jets" << std::endl;
    */

    // Use regular (not track) jets for counting
    bool is_in_higgs;
    bool is_in_W;
    if (debugMode) std::cout << "\t\t" << "Beginning loop over " << Jets.size() << " Track jets" << std::endl;
    if (debugMode) std::cout << "\t\t" << "Beginning loop over " << JetIsTaggedBin.size() << " Track jets" << std::endl;
    for (int i = 0; i < Jets.size(); i++) {
        if (JetIsTaggedBin.at(i) >= m_btagCategoryBin) {
            m_nTrkTagsOutside++;
            // is_in_higgs     = Higgs.DeltaR(Jets.at(i)) < Small_Jet_removal;
            // is_in_W         = W_hadronic.DeltaR(Jets.at(i)) < Small_Jet_removal;
            // if (!is_in_higgs) { // Not inside Higgs
            //     if (!is_in_W) { // Not inside Higgs or W
            //         m_nTrkTagsOutside++;
            //     }
            //     else { // Not inside Higgs but is inside W
            //         m_nTrkTagsInW++;
            //     }
            // }
            // else { // Is inside Higgs
            //     if (!is_in_W) { // Is inside higgs, not inside W
            //         m_nTrkTagsInH++;
            //     }
            //     else{ // Is inside Higgs & W
            //     // Choose whichever is closer
            //         if (Higgs.DeltaR(Jets.at(i)) < W_hadronic.DeltaR(Jets.at(i))) {
            //             m_nTrkTagsInH++;
            //         }
            //         else {
            //             m_nTrkTagsInW++;
            //         }
            //     }
            // }
        }
    }

    // Now do the same for the case of W constructed using SmallR and the case of W constructed using LargeR (for studies into redefining the categories, can probably be removed later)
    m_nTrkTagsOutside_smallR = 0;
    m_nTrkTagsInW_smallR = 0;
    m_nTrkTagsInH_smallR = 0;
    bool is_in_higgs_smallR;
    bool is_in_W_smallR;
    if (debugMode) std::cout << "\t\t" << "Beginning loop over " << Jets.size() << " Track jets" << std::endl;
    if (debugMode) std::cout << "\t\t" << "Beginning loop over " << JetIsTaggedBin.size() << " Track jets" << std::endl;
    for (int i = 0; i < Jets.size(); i++) {
        if (JetIsTaggedBin.at(i) >= m_btagCategoryBin) {
            m_nTrkTagsOutside_smallR++;
            // is_in_higgs_smallR     = Higgs.DeltaR(Jets.at(i)) < Small_Jet_removal;
            // is_in_W_smallR  = W_hadronic_FromSmallRJets.DeltaR(Jets.at(i)) < Small_Jet_removal;
            // if (!is_in_higgs_smallR) { // Not inside Higgs
            //     if (!is_in_W_smallR) { // Not inside Higgs or W
            //         m_nTrkTagsOutside_smallR++;
            //     }
            //     else { // Not inside Higgs but is inside W
            //         m_nTrkTagsInW_smallR++;
            //     }
            // }
            // else { // Is inside Higgs
            //     if (!is_in_W_smallR) { // Is inside higgs, not inside W
            //         m_nTrkTagsInH_smallR++;
            //     }
            //     else{ // Is inside Higgs & W
            //     // Choose whichever is closer
            //         if (Higgs.DeltaR(Jets.at(i)) < W_hadronic_FromSmallRJets.DeltaR(Jets.at(i))) {
            //             m_nTrkTagsInH_smallR++;
            //         }
            //         else {
            //             m_nTrkTagsInW_smallR++;
            //         }
            //     }
            // }
        }
    }
    m_nTrkTagsOutside_largeR = 0;
    m_nTrkTagsInW_largeR = 0;
    m_nTrkTagsInH_largeR = 0;
    bool is_in_higgs_largeR;
    bool is_in_W_largeR;
    if (debugMode) std::cout << "\t\t" << "Beginning loop over " << Jets.size() << " Track jets" << std::endl;
    if (debugMode) std::cout << "\t\t" << "Beginning loop over " << JetIsTaggedBin.size() << " Track jets" << std::endl;
    for (int i = 0; i < Jets.size(); i++) {
        if (JetIsTaggedBin.at(i) >= m_btagCategoryBin) {
            m_nTrkTagsOutside_largeR++;
            // is_in_higgs_largeR     = Higgs.DeltaR(Jets.at(i)) < Small_Jet_removal;
            // is_in_W_largeR  = W_hadronic_FromLargeRJet.DeltaR(Jets.at(i)) < Small_Jet_removal;
            // if (!is_in_higgs_largeR) { // Not inside Higgs
            //     if (!is_in_W_largeR) { // Not inside Higgs or W
            //         m_nTrkTagsOutside_largeR++;
            //     }
            //     else { // Not inside Higgs but is inside W
            //         m_nTrkTagsInW_largeR++;
            //     }
            // }
            // else { // Is inside Higgs
            //     if (!is_in_W_largeR) { // Is inside higgs, not inside W
            //         m_nTrkTagsInH_largeR++;
            //     }
            //     else{ // Is inside Higgs & W
            //     // Choose whichever is closer
            //         if (Higgs.DeltaR(Jets.at(i)) < W_hadronic_FromLargeRJet.DeltaR(Jets.at(i))) {
            //             m_nTrkTagsInH_largeR++;
            //         }
            //         else {
            //             m_nTrkTagsInW_largeR++;
            //         }
            //     }
            // }
        }
    }
    if (debugMode) std::cout << "\t\t" << "Leaving btagCounting" << std::endl;
}


// TODO should make the reference to vec2 a reference or a const or something to make sure it never gets changed
void EventLoop::Sort_vec1_by_vec2(std::vector<float> *vec1, std::vector<float> *vec2)
{
    if (debugMode) std::cout << "\t\t" << "Entering Sort_vec1_by_vec2" << std::endl;
    bool bDone = false;
    while (!bDone)
    {
        bDone = true;
        if (debugMode) std::cout << "\t\t\t" << "Beginning loop over " << vec2->size() << " jets" << std::endl;
        for (unsigned int i = 0; i < vec2->size() - 1 && vec2->size() > 0; i++)
        {
            if (vec2->at(i) < vec2->at(i + 1))
            {
                float tmp_val = vec1->at(i);
                vec1->at(i) = vec1->at(i + 1);
                vec1->at(i + 1) = tmp_val;
                bDone = false;
            }
        }
        if (debugMode) std::cout << "\t\t\t" << "Finished loop over jets" << std::endl;
    }
    if (debugMode) std::cout << "\t\t" << "Leaving Sort_vec1_by_vec2" << std::endl;
}


float EventLoop::SetNormFactor(float xsec, float kfac, float sumOfMCGenWeights){
    m_xsec = xsec;
    if (is_signal_sample) m_xsec = 1;
    //m_xsec = 1;
    m_kfac = kfac;
    m_sumOfMCGenWeights = sumOfMCGenWeights;
    weight_normalise = m_xsec*m_kfac/m_sumOfMCGenWeights;
    return weight_normalise;
}


float EventLoop::SetLumiFactor(float lumi_factor){
    luminosity_factor = lumi_factor;
    return luminosity_factor;
}










/* Unused in Boosted channel (currently)
bool EventLoop::PassEventSelectionResolved()
{
    if (MET->Pt() < 0.)
        return false;
    if (Leptons.at(0)->Pt() < 30000)
        return false;
    if (Lepton_Charge < 0)
    {
        if (m_NTags_caloJ < 2)
            return false;
        if (Jets.size() < 4)
            return false;
        ///if(Jets.size()   < 5)return false;
        if (FindJetPair_qqbb() == false)
            return false;
        ////if(Higgs.Pt()*0.001 < 100.)return false;
        ////if(Wplus.Pt()*0.001 < 100.)return false;
        return true;
    }
    else
    {
        if (m_NTags_caloJ < 2)
            return false;
        if (Jets.size() < 4)
            return false;
        if (FindJetPair_lvbb() == false)
            return false;
        if (Higgs.Pt() * 0.001 < 100.)
            return false;
        if (Wplus.Pt() * 0.001 < 120.)
            return false;
        return true;
    }
    return true;
}
*/

/* Unused in Boosted channel (currently)
void EventLoop::MatchTruthParticlesToJets()
{ // TODO Maybe comment out the occurances of FillMVATree if they are not used
    m_index_H1 = -99;
    m_index_H2 = -99;
    m_index_W1 = -99;
    m_index_W2 = -99;
    m_min_dRTruth = 999;
    if(!found_Higgs_constituents || !found_Wplus_had_constituents)return;
    bool status = false;
    Higgs_LV = (Higgs_p1 + Higgs_p2);
    Wplus_LV = (Wplus_p1 + Wplus_p2);
    if (Jets.size() < 4)
        return;
    for (unsigned int i = 0; i < Jets.size(); i++)
    {
        for (unsigned int j = i + 1; j < Jets.size(); j++)
        {
            for (unsigned int k = 0; k < Jets.size(); k++)
            {
                for (unsigned int l = k + 1; l < Jets.size(); l++)
                {
                    if (l == k || l == j || l == i || k == j || k == i || j == i)
                        continue;
                    TLorentzVector H = Jets.at(i) + Jets.at(j);
                    TLorentzVector W = Jets.at(k) + Jets.at(l);
                    double dRTruth = sqrt(pow(H.DeltaR(Higgs_LV), 2) + pow(W.DeltaR(Wplus_LV), 2));
                    if (m_min_dRTruth > dRTruth && H.DeltaR(Higgs_LV) < 0.3 && W.DeltaR(Wplus_LV) < 0.3)
                    {
                        m_min_dRTruth = dRTruth;
                        if (status == true)
                            FillMVATree(m_index_H1, m_index_H2, m_index_W1, m_index_W2, false);
                        m_index_H1 = i;
                        m_index_H2 = j;
                        m_index_W1 = k;
                        m_index_W2 = l;
                        status = true;
                    }
                    else
                    {
                        FillMVATree(i, j, k, l, false);
                    }
                }
            }
        }
    }
    if (status == true)
        FillMVATree(m_index_H1, m_index_H2, m_index_W1, m_index_W2, true);
}
*/

/* Unused in Boosted channel (currently)
void EventLoop::FillMVATree(int i_H1, int i_H2, int i_w1, int i_w2, bool is_signal)
{
    m_H_mass = (Jets.at(i_H1) + Jets.at(i_H2)).M() * 0.001;
    m_H_pT = (Jets.at(i_H1) + Jets.at(i_H2)).Pt() * 0.001;
    m_pTjH1 = Jets.at(i_H1).Pt() * 0.001;
    m_pTjH2 = Jets.at(i_H2).Pt() * 0.001;
    m_btagjH1 = (float)JetIsTagged.at(i_H1);
    m_btagjH2 = (float)JetIsTagged.at(i_H2);
    m_dRjjH = Jets.at(i_H1).DeltaR(Jets.at(i_H2));
    m_Wp_mass = (Jets.at(i_w1) + Jets.at(i_w2)).M() * 0.001;
    m_Wp_pT = (Jets.at(i_w1) + Jets.at(i_w2)).Pt() * 0.001;
    m_pTjWp1 = Jets.at(i_w1).Pt() * 0.001;
    m_pTjWp2 = Jets.at(i_w2).Pt() * 0.001;
    m_btagjWp1 = (float)JetIsTagged.at(i_w1);
    m_btagjWp2 = (float)JetIsTagged.at(i_w2);
    m_dRjjWp = Jets.at(i_H1).DeltaR(Jets.at(i_H2));
    m_Phi_HW = fabs((Jets.at(i_H1) + Jets.at(i_H2)).DeltaPhi((Jets.at(i_w1) + Jets.at(i_w2))));
    m_mass_VH = (Jets.at(i_H1) + Jets.at(i_H2) + Jets.at(i_w1) + Jets.at(i_w2)).M() * 0.001;
    m_is_Signal = is_signal;
    //m_myTree->Fill();
}
*/


/* Unused in Boosted channel (currently)
void EventLoop::initializeMVA_qqbb()
{
    m_reader_qqbb = new TMVA::Reader("!Color:!Silent");
    m_reader_qqbb->AddVariable("btagjH1", &m_btagjH1);
    m_reader_qqbb->AddVariable("btagjH2", &m_btagjH2);
    m_reader_qqbb->AddVariable("btagjW1", &m_btagjWp1);
    m_reader_qqbb->AddVariable("btagjW2", &m_btagjWp2);
    m_reader_qqbb->AddVariable("H_mass", &m_H_mass);
    m_reader_qqbb->AddVariable("Wp_mass", &m_Wp_mass);
    m_reader_qqbb->AddVariable("Phi_HW", &m_Phi_HW);
    m_reader_qqbb->AddVariable("H_pT/mass_VH", &m_pTH_over_mvH);
    m_reader_qqbb->AddVariable("Wp_pT/mass_VH", &m_ptW_over_mvH);
    // m_reader_qqbb->BookMVA("BDT", "dataset/weights/TMVAClassificationCategory_WpH_Tagger_v2.weights.xml");
}
*/

/* Unused in Boosted channel (currently)
void EventLoop::initializeMVA_lvbb()
{
    m_reader_lvbb = new TMVA::Reader("!Color:!Silent");
    m_reader_lvbb->AddVariable("btagjH1", &m_btagjH1);
    m_reader_lvbb->AddVariable("btagjH2", &m_btagjH2);
    m_reader_lvbb->AddVariable("H_mass", &m_H_mass);
    m_reader_lvbb->AddVariable("Phi_HW", &m_Phi_HW);
    m_reader_lvbb->AddVariable("pTHmvH", &m_pTH_over_mvH);
    m_reader_lvbb->AddVariable("pTWmvH", &m_ptW_over_mvH);
    // m_reader_lvbb->BookMVA("BDT", "dataset/weights/TMVAClassificationCategory_WpH_Tagger_lvbb.weights.xml");
}
*/

/* Unused in Boosted channel (currently)
double EventLoop::EvaluateMVAResponse_qqbb(int i_H1, int i_H2, int i_w1, int i_w2)  // unused, was for the resolved jets search
{
    m_H_mass = (Jets.at(i_H1) + Jets.at(i_H2)).M() * 0.001;
    m_Wp_mass = (Jets.at(i_w1) + Jets.at(i_w2)).M() * 0.001;
    m_btagjH1 = (float)JetIsTagged.at(i_H1);
    m_btagjH2 = (float)JetIsTagged.at(i_H2);
    m_btagjWp1 = (float)JetIsTagged.at(i_w1);
    m_btagjWp2 = (float)JetIsTagged.at(i_w2);
    m_Phi_HW = fabs((Jets.at(i_H1) + Jets.at(i_H2)).DeltaPhi((Jets.at(i_w1) + Jets.at(i_w2))));
    m_pTH_over_mvH = (Jets.at(i_H1) + Jets.at(i_H2)).Pt() / (Jets.at(i_H1) + Jets.at(i_H2) + Jets.at(i_w1) + Jets.at(i_w2)).M();
    m_ptW_over_mvH = (Jets.at(i_w1) + Jets.at(i_w2)).Pt() / (Jets.at(i_H1) + Jets.at(i_H2) + Jets.at(i_w1) + Jets.at(i_w2)).M();
    return m_reader_qqbb->EvaluateMVA("BDT");
}
*/

/* Unused in Boosted channel (currently)
double EventLoop::EvaluateMVAResponse_lvbb(int i, int j, TLorentzVector W)
{
    TLorentzVector H = Jets.at(i) + Jets.at(j);
    TLorentzVector vH = H + W;
    m_btagjH1 = (float)JetIsTagged.at(i);
    m_btagjH2 = (float)JetIsTagged.at(j);
    m_H_mass = (Jets.at(i) + Jets.at(j)).M() * 0.001;
    m_Phi_HW = fabs((Jets.at(i) + Jets.at(j)).DeltaPhi(W));
    m_pTH_over_mvH = H.Pt() / vH.M();
    m_ptW_over_mvH = W.Pt() / vH.M();
    return m_reader_lvbb->EvaluateMVA("BDT");
}
*/

/* Unused in Boosted channel (currently)
void EventLoop::WriteMVAInput()
{
    m_MET = MET->Pt() * 0.001;
    m_Lep_PT = Leptons.at(0)->Pt() * 0.001;
    m_Wleptonic_pT = Wminus.Pt() * 0.001;
    m_Wleptonic_Eta = fabs(Wminus.Eta());
    m_is_Signal = 1;
    //m_myTree->Fill();
}
*/

/* Unused in boosted channel at the moment
int EventLoop::GetTagWeightBin(double btag_score)
{
    if (btag_score < 0.11)
        return 0; /// 100% - 85%
    if (btag_score < 0.64)
        return 1; /// 85%  - 77%
    if (btag_score < 0.83)
        return 2; /// 77%  - 70%
    if (btag_score < 0.94)
        return 3; /// 70%  - 60%
    return 4;     /// 60%
}
*/


/* Unused in Boosted channel (currently)
bool EventLoop::FindJetPair_qqbb()  // unused, was for the resolved jets search
{
    m_index_H1 = -99;
    m_index_H2 = -99;
    m_index_W1 = -99;
    m_index_W2 = -99;
    bool status = false;
    m_MaxMVA_Response = -2;
    for (unsigned int i = 0; i < (Jets.size() < 6 ? Jets.size() : 6); i++)
    {
        for (unsigned int j = i + 1; j < (Jets.size() < 6 ? Jets.size() : 6); j++)
        {
            for (unsigned int k = 0; k < (Jets.size() < 6 ? Jets.size() : 6); k++)
            {
                for (unsigned int l = k + 1; l < (Jets.size() < 6 ? Jets.size() : 6); l++)
                {
                    if (l == k || l == j || l == i || k == j || k == i || j == i)
                        continue;
                    double mva_response = EvaluateMVAResponse_qqbb(i, j, k, l);
                    TLorentzVector H = Jets.at(i) + Jets.at(j);
                    TLorentzVector W = Jets.at(k) + Jets.at(l);
                    if (m_MaxMVA_Response < mva_response)
                    {
                        m_MaxMVA_Response = mva_response;
                        m_index_H1 = i;
                        m_index_H2 = j;
                        m_index_W1 = k;
                        m_index_W2 = l;
                        Wplus = Jets.at(m_index_W1) + Jets.at(m_index_W2);
                        Higgs = Jets.at(m_index_H1) + Jets.at(m_index_H2);
                        status = true;
                    }
                }
            }
        }
    }
    Set_Jet_observables();
    m_DeltaPhi_HW = fabs(Wplus.DeltaPhi(Higgs));
    m_mVH = (Wplus + Higgs).M() * 0.001;
    for (unsigned int i = 0; i < Jets.size(); i++)
    {
        if (i == m_index_H1 || i == m_index_H2 || i == m_index_W1 || i == m_index_W2)
            continue;
        if (JetIsTagged.at(i) >= m_btagCategoryBin)
            m_ntagsOutside++;
    }
    m_NTags_Higgs = (JetIsTagged.at(m_index_H1) >= m_btagCategoryBin) + (JetIsTagged.at(m_index_H2) >= m_btagCategoryBin);
    m_bTagCategory = GetBTagCategory(m_NTags_Higgs, m_ntagsOutside);
    m_NTags = GetBTagCategoryShort(m_NTags_Higgs, m_ntagsOutside);
    return status;
}
*/

/* Unused in Boosted channel (currently)
bool EventLoop::FindJetPair_lvbb()
{
    m_index_H1 = 1;
    m_index_H2 = 1;
    bool status = false;
    bool status_W = false;
    m_MaxMVA_Response = -2;
    for (int i = 0; i < Jets.size(); i++)
    {
        for (int j = i + 1; j < Jets.size(); j++)
        {
            if (i == j)
                continue;
            TLorentzVector W = GetWBoson(status_W);
            TLorentzVector H = Jets.at(i) + Jets.at(j);
            double mva_response_lvbb = EvaluateMVAResponse_lvbb(i, j, W);
            if (m_MaxMVA_Response < mva_response_lvbb)
            {
                m_MaxMVA_Response = mva_response_lvbb;
                m_index_H1 = i;
                m_index_H2 = j;
                Wplus = W;
                Higgs = H;
                status = true;
            }
        }
    }
    Set_Jet_observables();
    m_DeltaPhi_HW = fabs(Wplus.DeltaPhi(Higgs));
    m_mVH = (Wplus + Higgs).M() * 0.001;
    for (unsigned int i = 0; i < Jets.size(); i++)
    {
        if (i == m_index_H1 || i == m_index_H2)
            continue;
        if (JetIsTagged.at(i) >= m_btagCategoryBin)
            m_ntagsOutside++;
    }
    m_NTags_Higgs = (JetIsTagged.at(m_index_H1) >= m_btagCategoryBin) + (JetIsTagged.at(m_index_H2) >= m_btagCategoryBin);
    m_bTagCategory = GetBTagCategory(m_NTags_Higgs, m_ntagsOutside);
    m_NTags = m_NTags_caloJ;
    return status;
}
*/