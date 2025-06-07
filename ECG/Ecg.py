from skimage.io import imread
from skimage import color
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu, gaussian
from skimage.transform import resize
from numpy import asarray
from skimage.metrics import structural_similarity
from skimage import measure
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import joblib
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import os
from natsort import natsorted
from sklearn import linear_model, tree, ensemble
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

# Greedy algorithm for thresholding
def greedy_threshold(image):
    best_thresh = 0
    max_contrast = -1

    for t in np.linspace(0.01, 0.99, 100):  # Wider range, finer steps
        foreground = image < t
        background = image >= t

        if np.sum(foreground) == 0 or np.sum(background) == 0:
            continue

        contrast = abs(np.mean(image[foreground]) - np.mean(image[background]))

        if contrast > max_contrast:
            max_contrast = contrast
            best_thresh = t

    return best_thresh

class ECG:
	def  getImage(self,image):
		image=imread(image)
		return image

	def GrayImgae(self,image):
		image_gray = color.rgb2gray(image)
		image_gray=resize(image_gray,(1572,2213))
		return image_gray

	def DividingLeads(self,image):
		Lead_1 = image[300:600, 150:643] # Lead 1
		Lead_2 = image[300:600, 646:1135] # Lead aVR
		Lead_3 = image[300:600, 1140:1625] # Lead V1
		Lead_4 = image[300:600, 1630:2125] # Lead V4
		Lead_5 = image[600:900, 150:643] #Lead 2
		Lead_6 = image[600:900, 646:1135] # Lead aVL
		Lead_7 = image[600:900, 1140:1625] # Lead V2
		Lead_8 = image[600:900, 1630:2125] #Lead V5
		Lead_9 = image[900:1200, 150:643] # Lead 3
		Lead_10 = image[900:1200, 646:1135] # Lead aVF
		Lead_11 = image[900:1200, 1140:1625] # Lead V3
		Lead_12 = image[900:1200, 1630:2125] # Lead V6
		Lead_13 = image[1250:1480, 150:2125] # Long Lead
		Leads=[Lead_1,Lead_2,Lead_3,Lead_4,Lead_5,Lead_6,Lead_7,Lead_8,Lead_9,Lead_10,Lead_11,Lead_12,Lead_13]
		fig , ax = plt.subplots(4,3)
		fig.set_size_inches(10, 10)
		x_counter=0
		y_counter=0
		for x,y in enumerate(Leads[:len(Leads)-1]):
			if (x+1)%3==0:
				ax[x_counter][y_counter].imshow(y)
				ax[x_counter][y_counter].axis('off')
				ax[x_counter][y_counter].set_title("Leads {}".format(x+1))
				x_counter+=1
				y_counter=0
			else:
				ax[x_counter][y_counter].imshow(y)
				ax[x_counter][y_counter].axis('off')
				ax[x_counter][y_counter].set_title("Leads {}".format(x+1))
				y_counter+=1
		fig.savefig('Leads_1-12_figure.png')
		fig1 , ax1 = plt.subplots()
		fig1.set_size_inches(10, 10)
		ax1.imshow(Lead_13)
		ax1.set_title("Leads 13")
		ax1.axis('off')
		fig1.savefig('Long_Lead_13_figure.png')
		return Leads

	def PreprocessingLeads(self,Leads):
		fig2 , ax2 = plt.subplots(4,3)
		fig2.set_size_inches(10, 10)
		x_counter=0
		y_counter=0
		for x,y in enumerate(Leads[:len(Leads)-1]):
			grayscale = color.rgb2gray(y)
			blurred_image = gaussian(grayscale, sigma=1)

			otsu_thresh = threshold_otsu(blurred_image)
			binary_otsu = blurred_image < otsu_thresh

			greedy_thresh = greedy_threshold(blurred_image)
			binary_greedy = blurred_image < greedy_thresh

			plt.imsave(f'otsu_lead_{x+1}.png', binary_otsu, cmap='gray')
			plt.imsave(f'greedy_lead_{x+1}.png', binary_greedy, cmap='gray')

			binary_global = binary_otsu  # Use Otsu for model consistency
			binary_global = resize(binary_global, (300, 450))
			if (x+1)%3==0:
				ax2[x_counter][y_counter].imshow(binary_global,cmap="gray")
				ax2[x_counter][y_counter].axis('off')
				ax2[x_counter][y_counter].set_title("pre-processed Leads {} image".format(x+1))
				x_counter+=1
				y_counter=0
			else:
				ax2[x_counter][y_counter].imshow(binary_global,cmap="gray")
				ax2[x_counter][y_counter].axis('off')
				ax2[x_counter][y_counter].set_title("pre-processed Leads {} image".format(x+1))
				y_counter+=1
		fig2.savefig('Preprossed_Leads_1-12_figure.png')
		fig3 , ax3 = plt.subplots()
		fig3.set_size_inches(10, 10)
		grayscale = color.rgb2gray(Leads[-1])
		blurred_image = gaussian(grayscale, sigma=1)
		otsu_thresh = threshold_otsu(blurred_image)
		binary_global = blurred_image < otsu_thresh
		ax3.imshow(binary_global,cmap='gray')
		ax3.set_title("Leads 13")
		ax3.axis('off')
		fig3.savefig('Preprossed_Leads_13_figure.png')

	def SignalExtraction_Scaling(self,Leads):
		fig4 , ax4 = plt.subplots(4,3)
		x_counter=0
		y_counter=0
		for x,y in enumerate(Leads[:len(Leads)-1]):
			grayscale = color.rgb2gray(y)
			blurred_image = gaussian(grayscale, sigma=0.7)
			global_thresh = threshold_otsu(blurred_image)  # Use Otsu for consistency
			binary_global = blurred_image < global_thresh
			binary_global = resize(binary_global, (300, 450))
			contours = measure.find_contours(binary_global,0.8)
			contours_shape = sorted([x.shape for x in contours])[::-1][0:1]
			for contour in contours:
				if contour.shape in contours_shape:
					test = resize(contour, (255, 2))
			if (x+1)%3==0:
				ax4[x_counter][y_counter].invert_yaxis()
				ax4[x_counter][y_counter].plot(test[:, 1], test[:, 0],linewidth=1,color='black')
				ax4[x_counter][y_counter].axis('image')
				ax4[x_counter][y_counter].set_title("Contour {} image".format(x+1))
				x_counter+=1
				y_counter=0
			else:
				ax4[x_counter][y_counter].invert_yaxis()
				ax4[x_counter][y_counter].plot(test[:, 1], test[:, 0],linewidth=1,color='black')
				ax4[x_counter][y_counter].axis('image')
				ax4[x_counter][y_counter].set_title("Contour {} image".format(x+1))
				y_counter+=1
			lead_no=x
			scaler = MinMaxScaler()
			fit_transform_data = scaler.fit_transform(test)
			Normalized_Scaled=pd.DataFrame(fit_transform_data[:,0], columns = ['X'])
			Normalized_Scaled=Normalized_Scaled.T
			if (os.path.isfile('scaled_data_1D_{lead_no}.csv'.format(lead_no=lead_no+1))):
				Normalized_Scaled.to_csv('Scaled_1DLead_{lead_no}.csv'.format(lead_no=lead_no+1), mode='a',index=False)
			else:
				Normalized_Scaled.to_csv('Scaled_1DLead_{lead_no}.csv'.format(lead_no=lead_no+1),index=False)
		fig4.savefig('Contour_Leads_1-12_figure.png')

	def CombineConvert1Dsignal(self):
		test_final=pd.read_csv('Scaled_1DLead_1.csv')
		location= os.getcwd()
		print(location)
		for files in natsorted(os.listdir(location)):
			if files.endswith(".csv"):
				if files!='Scaled_1DLead_1.csv':
					df=pd.read_csv('{}'.format(files))
					test_final=pd.concat([test_final,df],axis=1,ignore_index=True)
		return test_final

	def DimensionalReduciton(self,test_final):
		pca_loaded_model = joblib.load('PCA_ECG (1).pkl')
		result = pca_loaded_model.transform(test_final)
		final_df = pd.DataFrame(result)
		return final_df

	def ModelLoad_predict(self,final_df):
		loaded_model = joblib.load('Heart_Disease_Prediction_using_ECG (4).pkl')
		result = loaded_model.predict(final_df)
		if result[0] == 1:
			return "You ECG corresponds to Myocardial Infarction"
		elif result[0] == 0:
			return "You ECG corresponds to Abnormal Heartbeat"
		elif result[0] == 2:
			return "Your ECG is Normal"
		else:
			return "You ECG corresponds to History of Myocardial Infarction"
