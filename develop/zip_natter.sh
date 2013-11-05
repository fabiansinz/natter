#!/bin/bash
cd ../../
zip -r natter/develop/natter-v0.2.zip natter/* -x natter/develop\* -x natter/build\* -x natter/natter.egg-info\* -x natter/.git\* natter/\*.ropeproject\* -x natter/\*.pyc
cd natter/develop

