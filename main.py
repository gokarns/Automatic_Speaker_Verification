import numpy as np
import boto3
import pymysql
import os
import awscli
from pathlib import Path
from itertools import groupby
from tqdm import tqdm
from sklearn import svm
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from resemblyzer import preprocess_wav, VoiceEncoder

from sklearn.model_selection import train_test_split
from sklearn import neighbors, datasets

from scipy.io.wavfile import read, write 

from Speaker_Verification import ASV

#Connect to AWS S3
s3 = boto3.client('s3', aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID'), aws_secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY'))
s3_resource = boto3.resource('s3')

#Connect to AWS RDS Databases
db = pymysql.connect(host = 'liveconnect.c8hkpfjqemqb.us-east-2.rds.amazonaws.com' , port = 3306, user = 'admin' , passwd = 'tSnH369!')

cursor = db.cursor()

# Select the LiveConnect Database
query = '''use LiveConnect'''
cursor.execute(query)

#aud = s3.download_file('bucket-live', 'f1/', 'name1-fwr-recording(1).wav')

#Select all rows from the user Table
query = '''select * from user'''
cursor.execute(query)

users = cursor.fetchall()
print(users)

asv = ASV(threshold=0.8)

#audio_file = 'audios/firetruck1.wav'

#audio_data = preprocess_wav(audio_file)


#voice_embed = asv.extract_features(audio_data)

#print(np.shape(voice_embed))
# print(audio_data)




