# Starbucks Capstone Challenge
This project is a submission for Udacity Data Scientist Nanodegree program.
A blog detailing the findings of this project can be found on https://lucas-guiraldelli.medium.com/starbucks-recommendation-of-offers-to-customers-challenge-1b828fe2c2ce

## Table of Contents
1. Required Installations
2. Dataset Descriptions
3. Results


## Installation
In order to facilitate the usability of the code, I used and prepared the environment so that the code can be used in Gitpod, all the configuration used is in the *.gitpod.yml* file

To run the code, install the dependencies used from the 'environment/requirements.txt' file.
To install the dependencies use the following command:

**pip install -r ././environment/requirements.txt**

NOTE: some libraries request python>=3.7, in this project I used python 3.8 by pattern of gitpod 


## Dataset overview
* The program used to create the data simulates how people make purchasing decisions and how those decisions are influenced by promotional offers.
* Each person in the simulation has some hidden traits that influence their purchasing patterns and are associated with their observable traits. People produce various events, including receiving offers, opening offers, and making purchases.
* As a simplification, there are no explicit products to track. Only the amounts of each transaction or offer are recorded.
* There are three types of offers that can be sent: buy-one-get-one (BOGO), discount, and informational. In a BOGO offer, a user needs to spend a certain amount to get a reward equal to that threshold amount. In a discount, a user gains a reward equal to a fraction of the amount spent. In an informational offer, there is no reward, but neither is there a requisite amount that the user is expected to spend. Offers can be delivered via multiple channels.
* The basic task is to use the data to identify which groups of people are most responsive to each type of offer, and how best to present each type of offer.

## Results
Based on our analysis we found that Starbucks customer demographic is middle aged to older folks with a annual income <$100000.
Discount offers have the maximum number of reactions from customers.

## Licensing, Authors, Acknowledgements
Dataset credits to Starbucks.
