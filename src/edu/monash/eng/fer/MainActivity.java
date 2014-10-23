//package org.opencv.samples.tutorial2;
package edu.monash.eng.fer;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import edu.monash.eng.fer.R;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;

import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.os.Bundle;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.MotionEvent;
import android.view.WindowManager;
import android.hardware.*;

public class MainActivity extends Activity implements CvCameraViewListener2 {
    private static final String    TAG = "OCVSample::Activity";
    private static final int       VIEW_MODE_NORMAL_MODE         = 0;
    private static final int       VIEW_MODE_LIVE_MODE         = 1;
    private static final int       VIEW_MODE_TRAINING_MODE     = 2;
    private static final int       VIEW_ABOUT                  = 3;
    private int                    mViewMode;
    private MenuItem               mItemNormalMode;
    private MenuItem               mItemLiveMode;
    private MenuItem               mItemTrainingMode;
    private MenuItem               mItemAbout;
    
    private File   				   mCascadeFileEyes;
    private File   				   mCascadeFileClassifier;
   
    private int					   mCameraId;
    private CameraBridgeViewBase   mOpenCvCameraView;
    
    private Mat                    mRgba, mRgbaF, mRgbaT;
    private Mat                    mGray;
    private int                    heightJava;
    private int                    widthJava;
    
    private int                    mModeFlag;
    private int                    mTouchFlag;
    private int                    mTrainingFinishFlag =0;
    private int                    trainingCounterJava;
    
    private File file;
    
    private BaseLoaderCallback  mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully");

                    // Load native library after(!) OpenCV initialization
                    System.loadLibrary("mixed_sample");
                    
                    mCameraId = 1;
                    mOpenCvCameraView.setCameraIndex(mCameraId);
                                
                    // here
                    loadClassifiers();
                    mOpenCvCameraView.enableView();
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

    public MainActivity() {
        Log.i(TAG, "Instantiated new " + this.getClass());
    }

    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.tutorial2_surface_view);
        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.tutorial2_activity_surface_view);
        mOpenCvCameraView.enableFpsMeter();
        mOpenCvCameraView.setCvCameraViewListener(this);
        

    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        Log.i(TAG, "called onCreateOptionsMenu");
        mItemNormalMode   = menu.add("Normal Mode");
        mItemLiveMode     = menu.add("Live Mode");
        mItemTrainingMode = menu.add("Training Mode");
        //mItemAbout = menu.add("About");
        return true;
    }

    @Override
    public void onPause()
    {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume()
    {
        super.onResume();
        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_3, this, mLoaderCallback);
    }

    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    public void onCameraViewStarted(int width, int height) {
    	mRgba = new Mat(height, width, CvType.CV_8UC3);// ori was 8uc4
        mGray = new Mat(height, width, CvType.CV_8UC1);
        heightJava  = height;
        widthJava   = width;
    	
    }

    public void onCameraViewStopped() {
        mRgba.release();
        mGray.release();
    }

    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
    	mRgba = inputFrame.rgba();
        mGray = inputFrame.gray();

        final int viewMode = mViewMode;

        switch (viewMode) {
        case VIEW_MODE_NORMAL_MODE:
        	Core.flip(mRgba, mRgba,1);
        	Core.putText(mRgba, "Normal Mode", new Point((widthJava/2)-200, 50) , 3 ,2,new Scalar(0,255,0),3);  
        	break;
        case VIEW_MODE_LIVE_MODE:
        	mModeFlag = 0 ;
            trainingCounterJava= performFER(mCascadeFileEyes.getAbsolutePath(),
        			 mCascadeFileClassifier.getAbsolutePath(),
        			 mGray.getNativeObjAddr(),
        			 mRgba.getNativeObjAddr(),
		             mModeFlag,
		             mTouchFlag);
            Log.d(TAG, "LiveModeCounter = " + trainingCounterJava);
        	break;
        case VIEW_MODE_TRAINING_MODE:
    		mModeFlag = 1 ;
    		
    		trainingCounterJava= performFER(mCascadeFileEyes.getAbsolutePath(),
        			   mCascadeFileClassifier.getAbsolutePath(),
        			   mGray.getNativeObjAddr(),
        			   mRgba.getNativeObjAddr(),
		               mModeFlag,
		               mTouchFlag
		               );
            Log.d(TAG, "outcome = " + trainingCounterJava);
            if (trainingCounterJava == 300){
            	trainingCounterJava = 0;
            	Log.i(TAG, "training completed");
	            String root = "/storage/emulated/0";
                File myDir = new File(root + "/finalYearProject");    
                myDir.mkdirs(); 
                file = new File (myDir,"svmClassifier2.yml");
                if (file.exists ()) file.delete (); 
				try {
					copy();
					
				} catch (IOException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
				//mViewMode = VIEW_MODE_LIVE_MODE;	
                }
            else{
            	Log.i(TAG, "training NOT completed");
            }
                
            break;
         }
        return mRgba;
    }
    
   
	@Override
    public boolean onTouchEvent(MotionEvent e) {
    	final int viewMode2 = mViewMode;
    	
    	if (e.getAction() == MotionEvent.ACTION_MOVE)
    	{
            	if (viewMode2 == VIEW_MODE_TRAINING_MODE){
            		Log.i(TAG, "Touch !");
            		if (mTouchFlag == 1) mTouchFlag = 0;
            		else if (mTouchFlag == 0) mTouchFlag = 1;
            	}
        }

        return true;
    }

    public void copy() throws IOException {
		InputStream in = new FileInputStream(mCascadeFileClassifier.getAbsolutePath());
		OutputStream out = new FileOutputStream(file);
		
		// Transfer bytes from in to out
		byte[] buf = new byte[1024];
		int len;
		while ((len = in.read(buf)) > 0) {
		    out.write(buf, 0, len);
		}
		in.close();
		out.close();
    }
    
    public void loadClassifiers(){
    	InputStream is; 
        FileOutputStream os;
       
        FileInputStream isClassifier;
        FileOutputStream osClassifier;
        
        try {   	
        	is = getResources().getAssets().open("lbpcascade_frontalface.xml");
            File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
            mCascadeFileEyes = new File(cascadeDir, "lbpcascade_frontalface.xml");
            
            //FileOutputStream os;
            os = new FileOutputStream(mCascadeFileEyes);
                    
            byte[] buffer = new byte[16384];
            int bytesRead;
            while ((bytesRead = is.read(buffer)) != -1) {
                os.write(buffer, 0, bytesRead);
            }

            is.close();
            os.close();
            Log.i(TAG, "face cascade found");
        } catch (IOException e) {
            Log.i(TAG, "face cascade not found");
        }
        
        try {
        	File file = new File("storage/emulated/0/finalYearProject/svmClassifier2.yml");          
            isClassifier = new FileInputStream(file);
            
        	
        	//isClassifier = getResources().getAssets().open("svmClassifier2.yml");
            File cascadeDirClassifier = getDir("cascade", Context.MODE_PRIVATE);
            mCascadeFileClassifier = new File(cascadeDirClassifier, "svmClassifier2.yml");     
            osClassifier = new FileOutputStream(mCascadeFileClassifier);
                    
            byte[] bufferClassifier = new byte[16384];
            int bytesReadClassifer;
            while ((bytesReadClassifer = isClassifier.read(bufferClassifier)) != -1) {
                osClassifier.write(bufferClassifier, 0, bytesReadClassifer);
            }

            isClassifier.close();
            osClassifier.close();
            Log.i(TAG, "svmClassifier found");
        } catch (IOException e) {
            Log.i(TAG, "svmClassifier not found");
        }

    }
    
    public boolean onOptionsItemSelected(MenuItem item) {
        Log.i(TAG, "called onOptionsItemSelected; selected item: " + item);
        if (item == mItemNormalMode){
        	mViewMode = VIEW_MODE_NORMAL_MODE;	
        }else
        if (item == mItemLiveMode) {
            mViewMode = VIEW_MODE_LIVE_MODE;
        }else 
        if (item == mItemTrainingMode){
        	mViewMode = VIEW_MODE_TRAINING_MODE;	
        }
        else 
            if (item == mItemAbout){
            	mViewMode = VIEW_ABOUT;	
            }
        
        return true;
    }
    
    public native int performFER(String EyesClassifier, 
         		                 String svmClassifier,
         		                 long matAddrGr,
         		                 long matAddrRgba,
         		                 int modeFlag,
         		                 int touchFlag);

}