#!/bin/bash



echo "------------------------- Installing Caffe -------------------------"
echo "NOTE: This script assumes that CUDA and cuDNN are already installed on your machine. Otherwise, it might fail."



function exitIfError {
    if [[ $? -ne 0 ]] ; then
        echo ""
        echo "------------------------- -------------------------"
        echo "Errors detected. Exiting script. The software might have not been successfully installed."
        echo "------------------------- -------------------------"
        exit 1
    fi
}



echo "------------------------- Checking Ubuntu Version -------------------------"
ubuntu_version="$(lsb_release -r)"
echo "Ubuntu $ubuntu_version"
if [[ $ubuntu_version == *"14."* ]]; then
    ubuntu_le_14=true
elif [[ $ubuntu_version == *"16."* || $ubuntu_version == *"15."* || $ubuntu_version == *"17."* || $ubuntu_version == *"18."* ]]; then
    ubuntu_le_14=false
else
    echo "Ubuntu release older than version 14. This installation script might fail."
    ubuntu_le_14=true
fi
exitIfError
echo "------------------------- Ubuntu Version Checked -------------------------"
echo ""



echo "------------------------- Checking Number of Processors -------------------------"
NUM_CORES=$(grep -c ^processor /proc/cpuinfo 2>/dev/null || sysctl -n hw.ncpu)
echo "$NUM_CORES cores"
exitIfError
echo "------------------------- Number of Processors Checked -------------------------"
echo ""



echo "------------------------- Compiling Caffe -------------------------"
cd autocompile/
if [[ $ubuntu_le_14 == true ]]; then
    cp Makefile.config.Ubuntu14.example ../Makefile.config
else
    cp Makefile.config.Ubuntu16.example ../Makefile.config
fi
cd ..
# make all -j$NUM_CORES
make all -j$NUM_CORES && make distribute -j$NUM_CORES
# make test -j$NUM_CORES
# make runtest -j$NUM_CORES
exitIfError
echo "------------------------- Caffe Compiled -------------------------"
echo ""



echo "------------------------- Caffe Installed -------------------------"
echo ""
