#!/bin/bash

TIME=10
ENV=quad
POINTS=100
STATE_SIZE=21

python main.py --train --episodes 100 --env ${ENV} --sim_time ${TIME} #--render_window

python main.py --resume ./training/Models/best.tar --env ${ENV} --eval --episodes 10 --sim_time ${TIME} #--render_window

python main.py --resume ./training/Models/best.tar --env ${ENV} \
               --export --gen_calib_data --points ${POINTS} --sim_time ${TIME}

freeze_graph --input_graph ./deploy/actor.pb \
    --input_checkpoint ./deploy/checkpoint.ckpt \
    --input_binary true \
    --output_graph ./deploy/frozen.pb \
    --output_node_names output_logits_actor_local/Identity


vai_q_tensorflow quantize \
    --input_frozen_graph ./deploy/frozen.pb \
    --input_fn input_func.calib_input \
    --output_dir ./deploy/quantized \
    --input_nodes input_data_actor_local \
    --output_nodes output_logits_actor_local/Identity \
    --input_shapes ?,1,1,${STATE_SIZE} \
    --calib_iter ${POINTS}  


python main.py --test_quantized --episodes 10 --env ${ENV} --sim_time ${TIME} #--render_window

vai_c_tensorflow \
    --frozen_pb ./deploy/quantized/deploy_model.pb \
    --arch /opt/vitis_ai/compiler/arch/dpuv2/Ultra96/Ultra96.json \
    --output_dir ./ultra96 \
    --net_name TD3PGagent

mv ultra96/dpu_TD3PGagent.elf ultra96/dpu_TD3PGagent_${ENV}.elf
rm dpu_TD3PGagent_${ENV}.elf
ln -s ./ultra96/dpu_TD3PGagent_${ENV}.elf dpu_TD3PGagent_${ENV}.elf

# On Ultra96
# sudo python3 -m ultra96.test_quad --episodes 10 --sim_time 10 --render_window --keyboard_pos
