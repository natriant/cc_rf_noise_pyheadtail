#!/bin/bash


for i in $(seq -20000.0 2000.0 20000.0); do
        echo ayy = $i;
        for j in {0..19}; do
		for k in {0..2}; do
                	cp sps_270GeV_PN1e-8_WakesOFF_QpxQpy5_ayy${i}_fixedKicksSet${j}_run${k}/file.txt /eos/user/n/natriant/pyheadtail_data/8Feb2021/tbt_data/QpxQpy5/WakesOFF/file_ayy${i}_set${j}_run${j}.txt
		done
        done
done


