# Automatic Identification of Novel Word Senses in Social Media
## About the Project
This project applies techniques based on word sense induction for determining instances of novel word senses. In the `get_data` file there are scripts for scraping social media data to provide the corpora for the project. 
## Requirements
The project is in Python, and requires a version of at least **3.6**. To install the required libraries, run
```
pip3 install -r requirements.txt
```
or
```
conda install --file requirements.txt
```
to use Anaconda
## Running the Project
There are two ways of running the project -- one for using [SemEval 2020 Task 1](https://competitions.codalab.org/competitions/20948) for evaluation, and one for the final project without a provided set of lemmas. Both projects require two textfiles, containing the two corpora which are to be compared, as well as the location for the final output, though the form of this output may differ based on which way the project is being run. To retrieve the social media data, there are two scripts in the `get_data.py` file. Calling the `scrape` script will produce a csv file with many millions of tweets in it. The `partition` script will take 500,000 of these tweets and produce two textfiles, one for the reference and one for the focus corpus. Running the project as part of SemEval requires an additional parameter, the location of a textfile containing the target lemmas. Run the project with
```
python main.py corpus1 corpus2 targets output --semeval_mode=True
```
when using for evaluation on SemEval, and with 
```
python main.py corpus1 corpus2 output
```
otherwise. Running the project in SemEval mode produces a file with the the binary classification of word sense change and a file with the Jensen-Shannon distance for all words in the file containing target words. Running the project for the final project produces a file with the *k* words identified as having changed the most, where *k* is a model hyperparameter with a default of 25.
