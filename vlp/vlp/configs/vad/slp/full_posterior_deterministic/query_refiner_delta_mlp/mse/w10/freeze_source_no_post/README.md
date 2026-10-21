This means that gradient is propagated from main loss -> model and also from slp loss only to SLP. 
In this case SLP module will learn simply to match some model features. --> could be later used for initialization 

No posterior inference, no grad flow to head otherwise! 