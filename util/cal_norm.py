import torch


def get_norm_of_lora(
    model,
    type="L2",
    group_num=6,
    group_type: str = "block",
    group_pos: str = "FFN",
    imagenet: bool = False,
):
    """
    get L2 norm of each group of lora parameters
    :param model: model (is already without ddp)
    :param type: L2 or L1
    :param group_num: 6 or 12 or 18
    :param group_type:
        -block (each Transformer block is a group)
        -lora (each LoRA is a group), 2 LoRAs in one block
        -matrix (each layer is a group), 2 matrix in one LoRA
    :param group_pos: lora pos
    :param imagenet: if imagenet, the model is vit_base_patch16_224
    :return: norm_list, list of norm of each group, type: list of tensor.float with length 12
    """
    with torch.no_grad():
        norm_list = []
        group_layers = []

        if group_pos == "FFN":
            if not imagenet:
                if group_type == "block":
                    for i in range(group_num):
                        group_item = []
                        group_item.append(
                            "transformer.layers.{}.1.fn.fn.net.0.lora_A".format(i)
                        )
                        group_item.append(
                            "transformer.layers.{}.1.fn.fn.net.0.lora_B".format(i)
                        )
                        group_item.append(
                            "transformer.layers.{}.1.fn.fn.net.3.lora_A".format(i)
                        )
                        group_item.append(
                            "transformer.layers.{}.1.fn.fn.net.3.lora_B".format(i)
                        )
                        group_layers.append(group_item)
                elif group_type == "lora":
                    for i in range(group_num):
                        group_item = []
                        group_item.append(
                            "transformer.layers.{}.1.fn.fn.net.0.lora_A".format(i)
                        )
                        group_item.append(
                            "transformer.layers.{}.1.fn.fn.net.0.lora_B".format(i)
                        )
                        group_layers.append(group_item)
                    for i in range(group_num):
                        group_item = []
                        group_item.append(
                            "transformer.layers.{}.1.fn.fn.net.3.lora_A".format(i)
                        )
                        group_item.append(
                            "transformer.layers.{}.1.fn.fn.net.3.lora_B".format(i)
                        )
                        group_layers.append(group_item)
                elif group_type == "matrix":
                    for i in range(group_num):
                        group_item = []
                        group_item.append(
                            "transformer.layers.{}.1.fn.fn.net.0.lora_A".format(i)
                        )
                        group_layers.append(group_item)
                    for i in range(group_num):
                        group_item = []
                        group_item.append(
                            "transformer.layers.{}.1.fn.fn.net.0.lora_B".format(i)
                        )
                        group_layers.append(group_item)
                    for i in range(group_num):
                        group_item = []
                        group_item.append(
                            "transformer.layers.{}.1.fn.fn.net.3.lora_A".format(i)
                        )
                        group_layers.append(group_item)
                    for i in range(group_num):
                        group_item = []
                        group_item.append(
                            "transformer.layers.{}.1.fn.fn.net.3.lora_B".format(i)
                        )
                        group_layers.append(group_item)
            else:  # imagenet is True, only consider the block grouping type
                for i in range(12):
                    group_item = []
                    group_item.append(
                        "encoder.layers.encoder_layer_{}.mlp.0.lora_A".format(i)
                    )
                    group_item.append(
                        "encoder.layers.encoder_layer_{}.mlp.0.lora_B".format(i)
                    )
                    group_item.append(
                        "encoder.layers.encoder_layer_{}.mlp.3.lora_A".format(i)
                    )
                    group_item.append(
                        "encoder.layers.encoder_layer_{}.mlp.3.lora_B".format(i)
                    )
                    group_layers.append(group_item)

        elif (
            group_pos == "Attention"
        ):  # NOTE: only for face tasks, not available for imagenet
            for i in range(group_num):
                group_item = []
                group_item.append(
                    "transformer.layers.{}.0.fn.fn.to_qkv.lora_A".format(i)
                )
                group_item.append(
                    "transformer.layers.{}.0.fn.fn.to_qkv.lora_B".format(i)
                )
                group_layers.append(group_item)

        print("\033[31mgroup_layers_names\033[0m\n", group_layers)
        # get the parameters
        group_params = []
        for group_item in group_layers:
            group_param = []
            for item in group_item:
                group_param.append(model.state_dict()[item])
            group_params.append(group_param)
        # print('group_parmas ', group_params)

        for group_param in group_params:
            if type == "L2":
                norm = 0
                length = len(group_param)
                for i in range(length):
                    norm += torch.norm(group_param[i])
                norm_list.append(norm)
            elif type == "L1":
                norm = 0
                length = len(group_param)
                for i in range(length):
                    norm += torch.norm(group_param[i], p=1)
                norm_list.append(norm)
            else:
                raise ValueError("type should be L1 or L2")
        return norm_list


"""
Imagenet layer names:
encoder.layers.encoder_layer_0.mlp.0
encoder.layers.encoder_layer_0.mlp.3
encoder.layers.encoder_layer_1.mlp.0
encoder.layers.encoder_layer_1.mlp.3
encoder.layers.encoder_layer_2.mlp.0
encoder.layers.encoder_layer_2.mlp.3
encoder.layers.encoder_layer_3.mlp.0
encoder.layers.encoder_layer_3.mlp.3
encoder.layers.encoder_layer_4.mlp.0
encoder.layers.encoder_layer_4.mlp.3
encoder.layers.encoder_layer_5.mlp.0
encoder.layers.encoder_layer_5.mlp.3
encoder.layers.encoder_layer_6.mlp.0
encoder.layers.encoder_layer_6.mlp.3
encoder.layers.encoder_layer_7.mlp.0
encoder.layers.encoder_layer_7.mlp.3
encoder.layers.encoder_layer_8.mlp.0
encoder.layers.encoder_layer_8.mlp.3
encoder.layers.encoder_layer_9.mlp.0
encoder.layers.encoder_layer_9.mlp.3
encoder.layers.encoder_layer_10.mlp.0
encoder.layers.encoder_layer_10.mlp.3
encoder.layers.encoder_layer_11.mlp.0
encoder.layers.encoder_layer_11.mlp.3
"""
