//package org.opencv.samples.tutorial2;
package edu.monash.eng.fer;
import edu.monash.eng.fer.R;
import android.os.Bundle;
import android.app.Activity;
import android.content.Intent;
 
public class splashScreen extends Activity {
 
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.splash_screen_main);
         
  
        	
        /****** Create Thread that will sleep for 5 seconds *************/        
        Thread background = new Thread() {
            public void run() {
                 
                try {
                    // Thread will sleep for 5 seconds
                    sleep(4*1000);
                     
                    // After 5 seconds redirect to another intent
                    Intent i=new Intent(getBaseContext(), MainActivity.class);//FirstScreen.class);
                    startActivity(i);
                     
                    //Remove activity
                    finish();
                     
                } catch (Exception e) {
                 
                }
            }
        };
         
        // start thread
        background.start();
         
//METHOD 2 
         
        /*
        new Handler().postDelayed(new Runnable() {
              
            // Using handler with postDelayed called runnable run method
  
            @Override
            public void run() {
                Intent i = new Intent(MainSplashScreen.this, FirstScreen.class);
                startActivity(i);
  
                // close this activity
                finish();
            }
        }, 5*1000); // wait for 5 seconds
        */
    }
     
    @Override
    protected void onDestroy() {
         
        super.onDestroy();
         
    }
}