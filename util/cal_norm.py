import torch


def get_norm_of_lora(model, type='L2'):
    """
    get L2 norm of each group of lora parameters
    :param model: model (is already without ddp)
    :param type: L2 or L1
    :return: norm_list, list of norm of each group, type: list of tensor.float with length 12
    """
    with torch.no_grad():
        norm_list = []
        group_layers = []
        for i in range(12):
            group_item = []
            if i < 6:
                group_item.append('transformer.encoder.layers.' + str(i) +
                                  '.linear1.lora_A')
                group_item.append('transformer.encoder.layers.' + str(i) +
                                  '.linear1.lora_B')
                group_item.append('transformer.encoder.layers.' + str(i) +
                                  '.linear2.lora_A')
                group_item.append('transformer.encoder.layers.' + str(i) +
                                  '.linear2.lora_B')
            else:
                group_item.append('transformer.decoder.layers.' + str(i - 6) +
                                  '.linear1.lora_A')
                group_item.append('transformer.decoder.layers.' + str(i - 6) +
                                  '.linear1.lora_B')
                group_item.append('transformer.decoder.layers.' + str(i - 6) +
                                  '.linear2.lora_A')
                group_item.append('transformer.decoder.layers.' + str(i - 6) +
                                  '.linear2.lora_B')
            group_layers.append(group_item)
        # print('group_layers_names', group_layers)
        # get the parameters
        group_params = []
        for group_item in group_layers:
            group_param = []
            for item in group_item:
                group_param.append(model.state_dict()[item])
            group_params.append(group_param)
        # print('group_parmas ', group_params)
        
        for group_param in group_params:
            if type == 'L2':
                norm_list.append(
                    torch.norm(group_param[0]) + torch.norm(group_param[1]) +
                    torch.norm(group_param[2]) + torch.norm(group_param[3]))
            elif type == 'L1':
                norm_list.append(
                    torch.norm(group_param[0], p=1) +
                    torch.norm(group_param[1], p=1) +
                    torch.norm(group_param[2], p=1) +
                    torch.norm(group_param[3], p=1))
            else:
                raise ValueError('type should be L1 or L2')
        return norm_list
