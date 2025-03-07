#!/usr/bin/env bash

INPUTS_DIR=/data/atlas/HplusWh/20241021_RawNtuples/*
# INPUTS_DIR=/eos/user/t/tqiu/H+Wh_ntuples/*
# Clear the output
rm nohup.out
start=$SECONDS

# Do the biggest file first to get the memory blocked out (I find this helps speed up for some reason)
# ./execute MCbase.config /eos/user/l/lubaines/ATLAS_SHARE/HpWh_L1ntuples/user.rhulsken.mc16_13TeV.407344.PhPy8EG_ttbarHT6c_1k_hdamp258p75_lep.TOPQ1.e6414s3126r10724p4512.Nominal_v0_1l_out_root/user.rhulsken.31923921._000118.out.root output3/

i=0
for sample_dir in $INPUTS_DIR
do
    if [[ $sample_dir == *"CORRUPTED"* ]]; then
        continue
    fi
    if [[ $sample_dir == *".save" ]]; then
        continue
    fi
    echo $sample_dir
    # # if [[ $sample_dir != *"410470"* ]]; then
    # if [[ $sample_dir != *"510122"* ]]; then
    if [[ $sample_dir != *"5101"* ]]; then
        continue
    fi
    # if [[ $sample_dir != *"410470"*"r9364"* ]]; then
    #     continue
    # fi
    # if [[ $sample_dir != *"mc16_13TeV.510117.MGPy8EGNNPDF30_Hp_H1000_Whbb.TOPQ1.e8448s3126r9364p4514."* ]]; then
    #     continue
    # fi
    if [[ $sample_dir == *"aMCNloP8_ttbarHT1k5_nonAH"* ]]; then # ATLFast samples to skip (will cause duplicate events if we include)
        continue
    fi
    if [[ $sample_dir == *"aMCNloP8_ttbarHT1k_1k5_nonAH"* ]]; then # ATLFast samples to skip (will cause duplicate events if we include)
        continue
    fi
    if [[ $sample_dir == *"PoPy8_WtDS_inclusive_t"* ]]; then # Diagram Subtraction samples to skip (will cause duplicate events of Diagram Removals if we include)
        continue
    fi
    if [[ $sample_dir == *"ttbar"* ]]; then # Can skip these ones if we only plan to use nominal ttbar samples
        if [[ $sample_dir != *"410470"* ]]; then
            continue
        fi
    fi
    # if [[ $sample_dir != *"ttbar"* ]]; then
    #     continue
    # fi
    # if [[ $sample_dir == *"410470"* ]]; then
    #     continue
    # fi
    # if [[ $sample_dir != *"ttbar"* ]]; then
    #     continue
    # fi
    # if [[ $sample_dir == *"410470"* ]]; then
    #     continue
    # fi
    # if [[ $sample_dir == *"ttbar"* ]]; then
    #     if [[ $sample_dir != *"410470"* ]]; then
    #         continue
    #     fi
    # fi
    # if [[ $sample_dir == *"mc16_13TeV.363"* ]]; then
    #     continue
    # fi
    # if [[ $sample_dir == *"mc16_13TeV.3"* ]]; then
    #     continue
    # fi
    # if [[ $sample_dir == *"mc16_13TeV.4"* ]]; then
    #     continue
    # fi
    # if [[ $sample_dir == *"mc16_13TeV.5"* ]]; then
    #     continue
    # fi
    # if [[ $sample_dir == *"mc16_13TeV.70032"* ]]; then
    #     continue
    # fi
    # if [[ $sample_dir == *"mc16_13TeV.700330"* ]]; then
    #     continue
    # fi
    # if [[ $sample_dir == *"mc16_13TeV.700331"* ]]; then
    #     continue
    # fi
    # if [[ $sample_dir == *"mc16_13TeV.700332"* ]]; then
    #     continue
    # fi
    # if [[ $sample_dir == *"mc16_13TeV.700333"* ]]; then
    #     continue
    # fi
    # if [[ $sample_dir != *"mc16_13TeV.510120"* ]]; then
    # if [[ $sample_dir != *"mc16_13TeV.510120.MGPy8EGNNPDF30_Hp_H1600_Whbb.TOPQ1.e8448s3126r9"* ]]; then
    # if [[ $sample_dir != *"mc16_13TeV.410470"* ]]; then
    # if [[ $sample_dir != *".mc16_13TeV.410470.PhPy8EG_ttbar_lep.TOPQ1.e6337s3126r10201p4514"* ]]; then
    # # if [[ $sample_dir != *"mc16_13TeV.510115.MGPy8EGNNPDF30_Hp_H800_Whbb.TOPQ1.e8448s3126r9364p4514"* ]]; then
    #     continue
    # fi
    echo "$i"
    # ./getNumEntriesExecute $file_name nominal_Loose
    ((i+=1))
    if [[ $i -gt 1000 ]]; then # Only do up to 1000 files
        continue
    fi
    # if [[ $i -gt 370 ]]; then # Don't do more than 20000 files because that's loads (more input than I currently have at the moment). Feel free to remove this
    #     continue
    # fi
    echo "Running on $sample_dir"
    bin/roo MCbase.cpp $sample_dir /data/atlas/HplusWh/20250305_WithTrueInclusion_FixedOverlapWHsjet/
    # echo "     "
done