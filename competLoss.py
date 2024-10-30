import torch
from modules import LobatschewskiFunktion



def get_compet_function(target_activation=0.1, scaling_rate=1.0, s_mul=0.5,compet_winner_weight=1.0, normalize=False, activation="relu", loss_weight = 1.0, **kwargs):
    if activation.lower() == 'lob':
        activation = LobatschewskiFunktion().cuda()
    elif activation.lower() == 'relu':
        activation = torch.nn.ReLU()
    elif activation.lower() == 'softmax':
        activation = torch.nn.Softmax()
    else:
        print(f"activation {activation} not supported")
        raise NotImplementedError()
        
    class compet_function(torch.autograd.Function):        
        @staticmethod
        def forward(ctx, logits, Y=None):
            activations = activation(logits)
            ctx.save_for_backward(activations)  
            result = activations             
            return result.sum()

        @staticmethod
        def backward(ctx, grad_output):
            ##print(grad_output)
            inputs, = ctx.saved_tensors   
            ##print("input shape:",input.shape )

            output = inputs - inputs.mean(dim=1, keepdim=True) - inputs.std(dim=1, keepdim=True) * .5
            output = torch.clamp(output, 0,10)
            output = torch.where(output==output.max(dim=-1, keepdim=True).values, -output * compet_winner_weight, output)
            #output = torch.clamp(input, -1,0)

            scaling_loss = (inputs-target_activation) * (inputs-target_activation)


            ##print(output.mean(), scaling.mean())
            
            if normalize:
                output =  scaling_loss *  output.mean() / scaling_loss.mean()

            ##print(output.mean(), scaling.mean())

            return loss_weight * (output + scaling_loss * scaling_rate ), None
        
    return compet_function
