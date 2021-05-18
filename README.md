# Job Recommendation System using Dynamic Weightage for implicit skill extraction

Build a job recommendation system that uses explicit and implicit skill extraction to extract skills from job description. Dynamic weightage is assigned to implicit skills to scale the impact factor.

## Install

This project requires **Python 3.0+** and uses the following libraries for different functions:

- NumPy - http://www.numpy.org/
- Pandas - http://pandas.pydata.org
- matplotlib - http://matplotlib.org/
- NLTK Stopwords - https://www.nltk.org/
- Selenium - https://www.seleniumhq.org/
- PyPDF2 - https://pythonhosted.org/PyPDF2/

## File structure

* jobRecommendationSystem.py - Main Python code containing the main() function
* config.py - Contains path for data used including job info and resume
* data - Directory containing the job info and sample resumes in PDF format
* FunctionsForJobRecommendation.py - Contains the different functions to extract skills and match them with resume

## Run

On the terminal window, navigate to the project directory and run the following

pip3 install nltk

pip3 install numpy

pip3 install matplotlib

pip3 install sklearn

pip3 PyPDF2

pip3 install pandas

python3 -m nltk.downloader stopwords

python3 jobRecommendationSystem.py



## Data
Data of the job descriptions is used from scraped data stored at :
https://raw.githubusercontent.com/wangpengcn/Job-recommendation-by-skill-match/master/data/indeed_jobs_info.json

