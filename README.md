# LearningJax

Earlier I didn't get any chance to experiment with JAX framwork. I was just knowing that it is an amazing thing that can accelerate Deep Learning by more **efficient numpy operation execution**, **Efficient Differentiation** and **Parallel Computation**. 

I come across this post on tweeter
![sayam_butani](/resources/images/sanyam_tweet.png)
and it motivated me to learn it everyday and take part in #27DaysofJax challenge.

I wanted to keep track of my learning in this repository.

## Goals:
- [ ] Complete JAX-101 Course
- [ ] Collect and Go through Amazing Blogs on JAX
- [ ] Create and Collect JAX Learning Resources

## Day1:

Earlier I was considering the JAX NumPy array and normal NumPy array as almost the same but there is a difference.
![day1_1](/resources/images/Day1_1.jpeg)

To Overcome this we can create a new copy of the JAX Numpy array like this,
![day1_2](/resources/images/Day1_2.png)

## Day2:

[Train Simple MLP in JAX Notebook](/Notebooks/TrainSimpleMLPwithJAX.ipynb)
![train_mlp](/resources/images/train_mlp_day2.png)

## Day3:

Compared Code for Linear Regression In Pytorch, Numpy and Jax

![LinearRegressionJax](/resources/images/LinearRegressionJax.png)
![LinearRegressionNumpy](/resources/images/LinearRegressionNumpy.png)
![LinearRegressionPytorch](/resources/images/LinearRegressionPytorch.png)

## Day4:
Started checking about [Flax](https://flax.readthedocs.io/en/latest/) A Neural Network based framework for JAX. We can train wide range of NLP and Vision based model on Huggingface library. Large Number of Pretrained models can be already found on Huggingface Model HUb.

## Day5
* The Proper way of Using JAX is to use it on **Functionally [Pure Python Function](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#pure-functions)**

## Day6
* Going through Jax101 content and understanding Jax better
* Its really interesting that in Jax we convert functions into `jaxpr` representation. `jaxpr` does not capture side effect function it only consider pure function. jaxpr captures the function as executed on the parameters given to it.
