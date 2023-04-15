#!/bin/bash

docker run -it --rm -p 8888:8888 -v "$(pwd):/app" tfm_carla_jupyter_env
