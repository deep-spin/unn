
# Forward only, zero initialization, unconstrained
for k in 2 3 4 5 
do
    python mnist_conv.py lr=0.0005 backward_loss_coef=0.0 unn_iter=$k epochs=50 constrained=False
done

# # Forward only, zero initialization
# for k in 1 2 3 4 5 
# do
#     python mnist_conv.py lr=0.0005 backward_loss_coef=0.0 unn_iter=$k epochs=50
# done

# # Forward-backward, zero initialization
# for k in 1 2 3 4 5 
# do
#     python mnist_conv.py lr=0.0005 backward_loss_coef=0.1 unn_iter=$k unn_order=fb epochs=50
# done


# # Forward only, random initialization
# for k in 1 2 3 4 5 
# do
#     python mnist_conv.py lr=0.0005 backward_loss_coef=0.0 unn_iter=$k unn_y_init=rand epochs=50
# done

# # Forward-backward, random initialization
# for k in 1 2 3 4 5 
# do
#     python mnist_conv.py lr=0.0005 backward_loss_coef=0.1 unn_iter=$k unn_y_init=rand unn_order=fb epochs=50
# done


# # Forward only, uniform initialization
# for k in 1 2 3 4 5 
# do
#     python mnist_conv.py lr=0.0005 backward_loss_coef=0.0 unn_iter=$k unn_y_init=uniform epochs=50
# done


# # Forward-backward, uniform initialization
# for k in 1 2 3 4 5 
# do
#     python mnist_conv.py lr=0.0005 backward_loss_coef=0.1 unn_iter=$k unn_y_init=uniform unn_order=fb epochs=50
# done