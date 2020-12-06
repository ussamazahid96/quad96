#!/bin/bash

sudo mkdir -p /opt/vitis_ai/compiler/arch/dpuv2/Ultra96
sudo cp -f Ultra96.json /opt/vitis_ai/compiler/arch/dpuv2/Ultra96/Ultra96.json
dlet -f dpu.hwh
sudo cp *.dcf /opt/vitis_ai/compiler/arch/dpuv2/Ultra96/Ultra96.dcf

