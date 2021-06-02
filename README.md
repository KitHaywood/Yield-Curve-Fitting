This repository requires datafiles off the following format, I will be setting up a MySQL server to allow access to sample data to allow this programme to run. The output of this is a graphical representation of fitted yield curve using Laguerre polynomials and for the time being remains a work in progress. 

Firstly - run the following to find the python version. This programme supports Python versions 3.8.2 onwards. 

```linux
python --version
```
If your version of python is previous to 3.8.2 then please update. 

Secondly, create a virtual environement in the command line by using:
```linux
python3 -m venv venv/
```
Followed by:
```linux
source venv/bin/activate
```
And finally install using:
```linux
pip3 install -r requirements.txt
```
Now we are ready to run! Make sure that you can connect to the database and that you have a working version of a MySQL client on your laptop. There are plenty of tutorials on how to set that up on youtube. 

The optimiser prints progress and output to the terminal, it is worth noting that for testing purposes using something like Colab is very useful because it can handle graphical objects as well as text and tqdm. 

This model also takes a brute force approach to optimisation by using permutations of assorted degrees of this brand of Orthogonal Polynomial. In the future I will try grid-search and basin hopping algos.
