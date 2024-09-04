#!/usr/bin/env bash

set -x
set -e

RESUME=$1
BASEDIR=$(dirname $1)

cp tools/export_to_onnx_flow.py $BASEDIR

pushd $BASEDIR

python export_to_onnx_flow.py $(basename $1)

popd
