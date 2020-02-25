# 2019/20 Computer Science Project
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
There are two ways of running the project -- one for using [SemEval 2020 Task 1](https://competitions.codalab.org/competitions/20948) for evaluation, and one for the final project without a provided set of lemmas. Both projects require two textfiles, containing the two corpora which are to be compared, as well as the location for the final output, though the form of this output may differ based on which way the project is being run. Running the project as part of SemEval requires an additional parameter, the location of a textfile containing the target lemmas. Run the project with
```
python main.py corpus1 corpus2 targets output --semeval_mode=True
```
when using for evaluation on SemEval, and with 
```
python main.py corpus1 corpus2 output
```
otherwise. 
