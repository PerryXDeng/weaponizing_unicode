# Puny Slayer
Puny Slayer is a machine learning solution for finding from 100,000 Unicodes characters that look alike,
which are used in homoglyph domain name attacks, also known as "punycode attacks," as well as social engineering.
For a detailed description of the attack, see https://blog.malwarebytes.com/101/2017/10/out-of-character-homograph-attacks-explained/. 
Here is Unicode Consortium's current recommendations on dealing with such attacks: http://www.unicode.org/reports/tr39/#Confusable\_Detection.
## Current Progress
We have implemented this variants of this one shot unsupervised learning 
[model](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf) in numpy, 
tensorflow core, and pytorch. We have selected the tensorflow core implementation 
due to its performance and flexibility with deployment. 

We are building the data pipeline needed to resample the unicode fonts in order to train and test the model.
## CommandLine Execution
Before you run anything, add the top-level folder to the Python path with the command:
export PYTHONPATH=$PYTHONPATH:/path/to/directoryroot
