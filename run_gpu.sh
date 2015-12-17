#!/bin/bash

THEANO_FLAGS=floatX=float32,device=gpu0 python tsne_theano.py
