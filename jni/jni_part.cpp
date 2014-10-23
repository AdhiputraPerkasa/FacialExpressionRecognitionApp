//-------------------------------------------------------------------------------------------
// FINAL YEAR PROJECT 2014
// ADHIPUTRA PERKASA
// 23169001
// MONASH UNIVERSITY MALAYSIA
//
// FACIAL EXPRESSION RECOGNITION (FER) IN ANDROID ENVIRONMENT
//
// This code is an implentation of Facial Expression Recognition (FER) previously implemented
// in C++ environment. The program has two modes, which are live and training mode. Training
// mode retrieves an image of type mat to detect facial images in a given frame, pre-process
// the image to stablize and balance lighting in Gaussian distribution, calculate the feature
// vector for every facial images by using Local Binary Pattern (LBP), and classify the raw
// expressions into n- number of expressions using Support Vector Machine (SVM), in which the
// parameters are previously optimised using k-fold method. In Live mode, the program retreives
// an input frame, detect facial areas, determine its feature vectors, and then use the previously
// trained SVM classifier to determine the given unknown expressions.
//
// This code is used in Java Native Interface (JNI) in android environment. You need to call this
// code with proper arguments, which is as follows:
//   1.
//   2.
//   3.
//   4.
//   5.
// This code is a part of partial fulfillment of final year project 2014 in Monash University
// Malaysia. Distributions of this code is open with approval of the Author and Monash University
// Malaysia.
//-------------------------------------------------------------------------------------------
// Include libraries required in this project
#include <jni.h>
#include <android/log.h>
#include <opencv2/core/core.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <time.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <opencv2/ml/ml.hpp>
#include <fcntl.h>
#include <errno.h>

// Declare definitions required in this project
#define  LOG_TAG    "FYP2014"
#define  LOGD(...)  __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define LBP_SIZE 32
#define NO_OF_LOCAL_REGIONS 16
#define BINS 256
#define CLIENT_NO 41

// Declare namespaces to avoid hassles later
using namespace std;
using namespace cv;

// EXTERN C STARTS HERE
extern "C" {
const char* EXPRESSION0 = "Neutral";
const char* EXPRESSION1 = "Happy";
const char* EXPRESSION2 = "Sad";
const char* EXPRESSION3 = "Surprise";
const char* EXPRESSION4 = "Anger";
const char* EXPRESSION5 = "Fear";
const char* EXPRESSION6 = "Disgust";

int FILE_PER_EXPRESSIONS = 50;
int EXPRESSIONSNO = 3;
int TOTALFILENEEDED = FILE_PER_EXPRESSIONS*EXPRESSIONSNO;

// FIRST STAGE  : FACE DETECTION GLOBAL VARIABLES
CascadeClassifier face_cascade, nose_cascade;
Mat _src,gray;
vector<Rect> faces, noses;
Mat featureVector, featureVector2, dst, hist;
Rect face;

// SECOND STAGE : LOCAL BINARY PATTERN GLOBAL VARIABLES
Mat crop, crop2, crop3;
Rect roi;
int xInt;

// THIRD STAGE  : SUPPORT VECTOR MACHINE GLOBAL CARIABLES
SVM svm;
int firstFlag, waitResponseCounter, prevResponse, response, localResponse; //Used in JK latches
Mat mother, labels;

// GENERAL GLOBAL VARIABLES
int x,y,i;
const char* jnamestrFaces;
const char* jnamestrClassifier;
int modeFlagInternal;
int touchFlagInternal;
int trainingCounter;
int firstTimeTouch = 0 ;
int currentTouchState, prevTouchState;
int currentLabel;
int svmReadytoTrainFlag;

// General Functions
void trainingMode();
void liveMode();
// First Stage: Face Detection
int faceDetection();
// Second Stage: Local Binary Pattern
void LBPStageAndSVM();
void executeLBP();
void LBP(Mat img);
void LBPhist(cv::Mat1b const& image);
// Third Stage: Support Vector Machines
void decision();
void prepareData();
void svmSettings();

// Function Prototypes for JNIEXPORT. Positively do not modify unless you are sure
//JNIEXPORT int JNICALL Java_org_opencv_samples_tutorial2_Tutorial2Activity_performFER
JNIEXPORT int JNICALL Java_edu_monash_eng_fer_MainActivity_performFER
(JNIEnv* jenv, jobject,jstring jfileNameFaces, jstring jClassifier, jlong addrGray, jlong addrRgba, jint modeFlag, jint touchFlag);

//JNIEXPORT int JNICALL Java_org_opencv_samples_tutorial2_Tutorial2Activity_performFER
JNIEXPORT int JNICALL Java_edu_monash_eng_fer_MainActivity_performFER
(JNIEnv* jenv, jobject,jstring jfileNameFaces, jstring jClassifier, jlong addrGray, jlong addrRgba, jint modeFlag, jint touchFlag)
{
	// Link Mat in JAVA by using JNI addressing
	_src = *(Mat*)addrRgba;
	gray = *(Mat*)addrGray;

	// Link jstring in JAVa using JNI addressing
	jnamestrFaces = jenv->GetStringUTFChars(jfileNameFaces, NULL);
	string stdFileNameFaces(jnamestrFaces);
	jnamestrClassifier = jenv->GetStringUTFChars(jClassifier, NULL);
	string stdFileNameClassifier(jnamestrFaces);
	modeFlagInternal  = (jint) modeFlag;
	touchFlagInternal = (jint) touchFlag;

	if (firstTimeTouch == 0){
		currentTouchState = touchFlagInternal;
		prevTouchState    = touchFlagInternal;
		firstTimeTouch    = -1;

		currentLabel=0;
	}
	else{
		prevTouchState    = currentTouchState;
		currentTouchState = touchFlagInternal;
	}



	// Debug: Check whether the svmClassifier is correct
	LOGD("jnamestrClassifier = %s", jnamestrClassifier);

	LOGD("Current Mode: %i", modeFlag);
	// Check modeFlag8
	if (modeFlagInternal == 0){
		// Load the svmClassifier
		// INFO: No need to check whether the file exist; it has been done in JAVA
		svm.load(jnamestrClassifier);

		// Load the cascade classifier for faces
		if (!face_cascade.load(stdFileNameFaces)){
			LOGD("LOADING CASCADE: lbpcascade_frontalface.xml loaded");
		}
		else{
			LOGD("LOADING CASCADE: lbpcascade_frontalface.xml FAILED to load");
		}
		liveMode();
		return -2;
	}
	else{
		trainingMode();
		return trainingCounter;
	}
}
void liveMode(){
	// INFO: Nose classifier is not used in this program
	// Flip both input matrices (rGba and grayscale) to achieve mirror-like effects
    flip(_src,_src,1);
    flip(gray,gray,1);

    // Start First Stage: Face-detection
	if (faceDetection()==1){
		LOGD("FACE: Face detected");

		// Start Second Stage: Local Binary Patterm
		LBPStageAndSVM();

		// Start Third Stage: Predicting outcome response using Support Vector Machine
		decision();
	}
	else if (faceDetection()==-1){
		LOGD("FACE: Face NOT detected");
		// Carry on..
	}

}

void trainingMode(){
	stringstream ssfnTrainingMode;
	String TrainingModeString = "";
	stringstream ssfnTrainingMode2;
	String TrainingModeString2 = "";



	// INFO: Nose classifier is not used in this program
	// Flip both input matrices (rGba and grayscale) to achieve mirror-like effects
    flip(_src,_src,1);
    flip(gray,gray,1);

    // Start First Stage: Face-detection
	if (faceDetection()==1){
		LOGD("FACE: Face detected");

		// need to notify user
		// once the posedge clk is triggered, raise a flag and put down previous flag
		// start lbp
		// once done, need to notify user is triggered and lala
		putText(_src,"Expression needed:"         , Point(10,90), FONT_HERSHEY_PLAIN , 4 , cvScalar(0, 0, 255), 5, CV_AA);
		if (trainingCounter >=   0 && trainingCounter <FILE_PER_EXPRESSIONS){
			putText(_src,EXPRESSION0 , Point(700,90), FONT_HERSHEY_PLAIN , 4 , cvScalar(0, 0, 255), 5, CV_AA);
		}else
		if (trainingCounter >= 1*FILE_PER_EXPRESSIONS && trainingCounter <2*FILE_PER_EXPRESSIONS){
			putText(_src,EXPRESSION1 , Point(700,90), FONT_HERSHEY_PLAIN , 4 , cvScalar(0, 0, 255), 5, CV_AA);
		}else
		if (trainingCounter >= 2*FILE_PER_EXPRESSIONS && trainingCounter <3*FILE_PER_EXPRESSIONS){
			putText(_src,EXPRESSION2 , Point(700,90), FONT_HERSHEY_PLAIN , 4 , cvScalar(0, 0, 255), 5, CV_AA);
		}else
		if (trainingCounter >= 3*FILE_PER_EXPRESSIONS && trainingCounter <4*FILE_PER_EXPRESSIONS){
			putText(_src,EXPRESSION3 , Point(700,90), FONT_HERSHEY_PLAIN , 4 , cvScalar(0, 0, 255), 5, CV_AA);
		}else
		if (trainingCounter >= 4*FILE_PER_EXPRESSIONS && trainingCounter <5*FILE_PER_EXPRESSIONS){
			putText(_src,EXPRESSION4 , Point(700,90), FONT_HERSHEY_PLAIN , 4 , cvScalar(0, 0, 255), 5, CV_AA);
		}else
		if (trainingCounter >= 5*FILE_PER_EXPRESSIONS && trainingCounter <6*FILE_PER_EXPRESSIONS){
			putText(_src,EXPRESSION5 , Point(700,90), FONT_HERSHEY_PLAIN , 4 , cvScalar(0, 0, 255), 5, CV_AA);
		}else
		if (trainingCounter >= 6*FILE_PER_EXPRESSIONS && trainingCounter <7*FILE_PER_EXPRESSIONS){
			putText(_src,EXPRESSION6 , Point(700,90), FONT_HERSHEY_PLAIN , 4 , cvScalar(0, 0, 255), 5, CV_AA);
		}

		if (trainingCounter != TOTALFILENEEDED){
			if (trainingCounter%FILE_PER_EXPRESSIONS==0){
				// need to notify user
				LOGD("PREV: %i | CURR: %i ", prevTouchState,currentTouchState);
				if(currentTouchState == prevTouchState){
					//sprintf(prepareDataString,"(Training Mode) Prepare facial expressions for: %s \n", EXPRESSION0);
					putText(_src,"TOUCH ANYWHERE TO CONTINUE" , Point (500,500), FONT_HERSHEY_PLAIN , 4 , cvScalar(0, 0, 255), 5, CV_AA);
					}
				else{
					if (trainingCounter == 0){
						currentLabel =0;
						svmReadytoTrainFlag = 1;
					}
					else{
						currentLabel++;
						mother.push_back(featureVector);
						labels.push_back(currentLabel);
						LOGD("RET: (%i), %i, %i,%i, %i (%i)", trainingCounter, mother.rows, mother.cols, labels.rows, labels.cols,  currentLabel);

					}
					trainingCounter++;
					putText(_src,"LOOK FAST" , Point (500,500), FONT_HERSHEY_PLAIN , 4 , cvScalar(0, 0, 255), 5, CV_AA);
				}
			}

			else
			{

				// Start Second Stage: Local Binary Patterm
				LBPStageAndSVM();

				mother.push_back(featureVector);
				labels.push_back(currentLabel);
				LOGD("RET: (%i), %i, %i,%i, %i (%i)", trainingCounter, mother.rows, mother.cols, labels.rows, labels.cols,  currentLabel);

				//LOGD("(%i) MOTHER : %i, %i", trainingCounter, mother.rows, mother.cols);
				if (trainingCounter%3 == 0){
					ssfnTrainingMode  << "Please wait.  ";
					ssfnTrainingMode2 << (trainingCounter*(100/FILE_PER_EXPRESSIONS))%100 << "%";
				}else
				if (trainingCounter%3 == 1){
					ssfnTrainingMode  << "Please wait.. ";
					ssfnTrainingMode2 << (trainingCounter*(100/FILE_PER_EXPRESSIONS))%100<< "%";
				}else
				if (trainingCounter%3 == 2){
					ssfnTrainingMode  << "Please wait...";
					ssfnTrainingMode2 << (trainingCounter*(100/FILE_PER_EXPRESSIONS))%100<< "%";
				}

				TrainingModeString  = ssfnTrainingMode.str();
				TrainingModeString2 = ssfnTrainingMode2.str();

				putText(_src, TrainingModeString , Point  (10,500), FONT_HERSHEY_PLAIN , 4 , cvScalar(0, 0, 255), 5, CV_AA);
				putText(_src, TrainingModeString2 , Point (10,600), FONT_HERSHEY_PLAIN , 4 , cvScalar(0, 0, 255), 5, CV_AA);

				LOGD("Labels: %i ", currentLabel);



				LOGD("(%i) featureVector (Training): %i, %i", trainingCounter, featureVector.rows, featureVector.cols);
				trainingCounter++;

			}
		}
		else{
			putText(_src,"Please wait while SVM is being trained" , Point (500,500), FONT_HERSHEY_PLAIN , 4 , cvScalar(0, 0, 255), 5, CV_AA);
			TrainingModeString2 = "";
			ssfnTrainingMode2 << trainingCounter << "Face images is being processed";
			TrainingModeString2 = ssfnTrainingMode2.str();
			putText(_src, TrainingModeString2 , Point (500,600), FONT_HERSHEY_PLAIN , 4 , cvScalar(0, 0, 255), 5, CV_AA);
			svmSettings();


		}

		// Start Third Stage: Predicting outcome response using Support Vector Machine
		//postSVM();
	}
	else if (faceDetection()==-1){
		LOGD("FACE: Face NOT detected");
		// Carry on..
	}
}


void svmSettings(){
	SVMParams params = SVMParams();
	params.svm_type = SVM::C_SVC;
	params.kernel_type = SVM::LINEAR;
	params.degree = 3.43; // for poly
	params.gamma = 0.00225; // for poly / rbf / sigmoid
	params.coef0 = 19.6; // for poly / sigmoid
	params.C = 0.5; // for CV_SVM_C_SVC , CV_SVM_EPS_SVR and CV_SVM_NU_SVR
	params.nu = 0.0; // for CV_SVM_NU_SVC , CV_SVM_ONE_CLASS , and CV_SVM_NU_SVR
	params.p = 0.0; // for CV_SVM_EPS_SVR
	params.class_weights = NULL; // for CV_SVM_C_SVC
	params.term_crit.type = CV_TERMCRIT_ITER | CV_TERMCRIT_EPS;
	params.term_crit.max_iter = 1000;
	params.term_crit.epsilon = 1e-6;

	if (svmReadytoTrainFlag == 1){
		svm.train(mother,labels,Mat(),Mat(),params);
	    svmReadytoTrainFlag = 0;
	    svm.save(jnamestrClassifier);
	    trainingCounter =0;
	}

}





// ----------------------------------------------------------------------------------------------------------------------
// WARNING: THE METHODS BELOW ARE POSITVELY CORRECT. DO NOT TRY TO MODIFY THESE METHODS UNLESS YOU ARE ABSOLUTELY SURE
// ----------------------------------------------------------------------------------------------------------------------
int faceDetection(){

	equalizeHist(gray, gray);
	face_cascade.detectMultiScale(gray, faces, 1.1, 3,  CV_HAAR_FIND_BIGGEST_OBJECT , Size(500, 500));

	if (faces.size() == 1){
        //TODO: ISN'T LINE BELOW REDUNDANT ?
		face = faces[0];
        face.x       = face.x + (face.width/6.3);
        face.y       = face.y + (face.height/6.1);
        face.width   = (face.x+(face.width/1.2))-(face.x + (face.width/6.3));
        face.height  = (face.y+(face.height/1.1))-(face.y + (face.height/6));
        rectangle(_src, face, Scalar(0, 255, 0), 5);
        crop = gray(face);
        return 1;
	}
	else{
		return -1;
	}

}


void LBPStageAndSVM(){
	// Initializing local variables

	// Releasing crop2
	crop2.release();

	// Resizing to initial 128 x 128 pixels before detecting nose
	resize(crop, crop2, Size(LBP_SIZE, LBP_SIZE), 0, 0, INTER_LINEAR);
	LOGD("Crop2 : %i , %i \n", crop2.rows, crop2.cols);

	// Proceed with LBP
	executeLBP();
}
void executeLBP(){
	featureVector.release();
	// Local Binary Pattern starts here
	xInt = LBP_SIZE/sqrt(double(NO_OF_LOCAL_REGIONS));
	for (y=0;y<LBP_SIZE;y+=xInt){
			for (x=0;x<LBP_SIZE;x+=xInt){
				roi.x = x; roi.y = y;
				roi.width = xInt;
				roi.height = xInt;
				crop3=crop2(roi);
				LOGD("Crop3 : %i , %i \n", crop3.rows, crop3.cols);
				LBP(crop3);

				LBPhist(dst);
				featureVector.push_back(hist);
				}
		}
	transpose(featureVector,featureVector);
	LOGD("featureVector : %i , %i \n", featureVector.rows, featureVector.cols);

}
void LBP(Mat img){
	dst = Mat::zeros(img.rows-2, img.cols-2, CV_8UC1);
	for(int i=1;i<img.rows-1;i++) {
		for(int j=1;j<img.cols-1;j++) {
			uchar center = img.at<uchar>(i,j);
			unsigned char code = 0;
			code |= ((img.at<uchar>(i-1,j-1)) > center) << 7;
			code |= ((img.at<uchar>(i-1,j)) > center) << 6;
			code |= ((img.at<uchar>(i-1,j+1)) > center) << 5;
			code |= ((img.at<uchar>(i,j+1)) > center) << 4;
			code |= ((img.at<uchar>(i+1,j+1)) > center) << 3;
			code |= ((img.at<uchar>(i+1,j)) > center) << 2;
			code |= ((img.at<uchar>(i+1,j-1)) > center) << 1;
			code |= ((img.at<uchar>(i,j-1)) > center) << 0;
			dst.at<uchar>(i-1,j-1) = code;
		}
	}
}
void LBPhist(cv::Mat1b const& image)
{
    // Set histogram bins count
    int bins = BINS;
    int histSize[] = {bins};
    // Set ranges for histogram bins
    float lranges[] = {0, 256};
    const float* ranges[] = {lranges};
    // create matrix for histogram
    int channels[] = {0};

    // create matrix for histogram visualization
    int const hist_height = 256;
    cv::Mat3b hist_image = cv::Mat3b::zeros(hist_height, bins);
	cv::calcHist(&image, 1, channels, cv::Mat(), hist, 1, histSize, ranges, true, false);
	double max_val=0;
    minMaxLoc(hist, 0, &max_val);
	//hist.size();
    //printf("Rows : %i | Columns = %i \n", hist.rows, hist.cols);
	// visualize each bin
    for(int b = 0; b < bins; b++) {
        float const binVal = hist.at<float>(b);
        int   const height = cvRound(binVal*hist_height/max_val);
        cv::line
            ( hist_image
            , cv::Point(b, hist_height-height), cv::Point(b, hist_height)
            , cv::Scalar::all(255)
            );
    }
}
void decision(){
	response = svm.predict(featureVector);


	if(firstFlag == 0){
		firstFlag = 1;
		waitResponseCounter = 0;
		prevResponse = response;
	}
	else if (firstFlag ==1){
		if(prevResponse == response){
			waitResponseCounter++;
			prevResponse = response;
		}
		else{
			waitResponseCounter =0;
			prevResponse = response;
		}
		if (waitResponseCounter == 4){
			localResponse = response;
			firstFlag = 0;
		}
	}

	LOGD("Response = %i", localResponse);



	if (localResponse == 0){
		putText(_src, EXPRESSION0  , Point((face.x+face.width/8), face.y-10), CV_FONT_HERSHEY_COMPLEX , 1.25, cvScalar(0, 255, 0), 2, CV_AA);
		LOGD("RESPONSE: %s",EXPRESSION0);

	}else
	if (localResponse == 1){
		putText(_src, EXPRESSION1  , Point((face.x+face.width/8), face.y-10), CV_FONT_HERSHEY_COMPLEX , 1.25, cvScalar(0, 255, 0), 2, CV_AA);
		LOGD("RESPONSE: %s",EXPRESSION1);
		putText(_src, "Set Brighter Light" , Point((_src.cols/2)-200, 50), FONT_HERSHEY_PLAIN , 4 , cvScalar(0, 255, 0), 5, CV_AA);
	}else
	if (localResponse == 2){
		putText(_src, EXPRESSION2  , Point((face.x+face.width/8), face.y-10), CV_FONT_HERSHEY_COMPLEX , 1.25, cvScalar(0, 255, 0), 2, CV_AA);
		LOGD("RESPONSE: %s",EXPRESSION2);
		putText(_src, "Set Dimmer Light" , Point((_src.cols/2)-200, 50), FONT_HERSHEY_PLAIN , 4 , cvScalar(0, 0, 255), 5, CV_AA);
	}else
	if (localResponse == 3){
		putText(_src, EXPRESSION3  , Point((face.x+face.width/8), face.y-10), CV_FONT_HERSHEY_COMPLEX , 1.25, cvScalar(0, 255, 0), 2, CV_AA);
		LOGD("RESPONSE: %s",EXPRESSION3);
	}else
	if (localResponse == 4){
		putText(_src, EXPRESSION4  , Point((face.x+face.width/8), face.y-10), CV_FONT_HERSHEY_COMPLEX , 1.25, cvScalar(0, 255, 0), 2, CV_AA);
		LOGD("RESPONSE: %s",EXPRESSION4);
	}else
	if (localResponse == 5){
		putText(_src, EXPRESSION5  , Point((face.x+face.width/8), face.y-10), CV_FONT_HERSHEY_COMPLEX , 1.25, cvScalar(0, 255, 0), 2, CV_AA);
		LOGD("RESPONSE: %s",EXPRESSION5);
	}else
	if (localResponse == 6){
		putText(_src, EXPRESSION6  , Point((face.x+face.width/8), face.y-10), CV_FONT_HERSHEY_COMPLEX , 1.25, cvScalar(0, 255, 0), 2, CV_AA);
		LOGD("RESPONSE: %s",EXPRESSION6);
	}






}
} //extern C
