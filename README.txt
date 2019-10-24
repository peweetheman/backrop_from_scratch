Patrick Phillips
CSC 246
Multilayer Perceptron

I first completed the basic framework for the neural network.
Since there were only two layer I did not loop through layers in backprop assign deltas etc.

In my code I use the development data to loop through and choose which set of weights over all iterations was optimal, 
this way avoiding overfitting the training data.
Typically as the iterations increased the performance continued to increase, although the neural net is definitely 
slower than a single perceptron so I did not try more than 25 iterations, but I found that after about 10 iterations
there was little or no improvement in the dev data, suggesting that the weights had converged to optimal values already.
I also experimented with different learn rates and had the most success with consistent and quick convergence to an accuracy of about
85% when I used a lr of .1, I found that sometimes the success % would get stuck at the .7567 accuracy with different combinations of hyperparameters,
which I was confuzed by because since I am doing updates on every data point I would think that getting stuck in local maxima would be impossible. This 
only happened occassionally however, so it may have just been a bug in code or weird pattern interaction with training data. However, with high learning 
rates over .01 this problem didn't occur, or I assume with very large number of iterations this problem won't occur. I found that overall, the neural 
network was better than the perceptron. As shown in the graph that I provided I had results that peaked at roughly 85% accuracy which if I recall is 
about 5% better than the perceptron was able to do.

