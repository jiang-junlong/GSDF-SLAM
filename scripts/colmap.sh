#!/bin/bash

../bin/train_colmap \
    ../cfg/colmap/gaussian_splatting.yaml \
    ../dataset/tandt_db/db/drjohnson \
    ../result/colmap/drjohnson \
    # no_viewer