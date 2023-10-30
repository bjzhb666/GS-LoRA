import torch


def get_norm_of_lora(model, type='L2', group_num=6):
    """
    get L2 norm of each group of lora parameters
    :param model: model (is already without ddp)
    :param type: L2 or L1
    :return: norm_list, list of norm of each group, type: list of tensor.float with length 12
    """
    with torch.no_grad():
        norm_list = []
        group_layers = []
        for i in range(group_num):
            group_item = []
            group_item.append('transformer.layers.{}.1.fn.fn.net.0.lora_A'.format(i))
            group_item.append('transformer.layers.{}.1.fn.fn.net.0.lora_B'.format(i))
            group_item.append('transformer.layers.{}.1.fn.fn.net.3.lora_A'.format(i))
            group_item.append('transformer.layers.{}.1.fn.fn.net.3.lora_B'.format(i))
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
