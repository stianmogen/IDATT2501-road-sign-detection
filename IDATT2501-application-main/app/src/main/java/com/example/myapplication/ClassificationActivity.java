package com.example.myapplication;

import static org.opencv.imgproc.Imgproc.cvtColor;

import android.Manifest;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.location.Location;
import android.location.LocationListener;
import android.location.LocationManager;
import android.os.Bundle;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.MotionEvent;
import android.view.WindowManager;
import android.widget.TextView;

import androidx.appcompat.widget.Toolbar;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.CLAHE;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.LinkedList;
import java.util.List;
import java.util.Objects;

/**
 * Classification Activity
 *
 * Implements CvCameraViewListener2 for real time classification from camera
 * Implements LocationListener for mobile unit location updates
 */
public class ClassificationActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2, LocationListener {

    private static final String TAG = "OCVSample::Activity";
    private static final Scalar OBJECT_RECT_COLOR = new Scalar(0, 255, 0);

    private LocationManager locationManager;
    private String geoLocation;
    
    private Mat mRgba;
    private Mat mGray;
    private File mCascadeFile;
    private CascadeClassifier cascadeDetector;
    private Module module;


    private MyCameraView mOpenCvCameraView;


    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {

        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    Log.i(TAG, "OpenCV Loaded Successfully");

                    mOpenCvCameraView.enableFpsMeter();
                    mOpenCvCameraView.setCameraIndex(0);
                    mOpenCvCameraView.enableView();
                }
                break;
                default: {
                    super.onManagerConnected(status);
                }
                break;
            }
        }
    };


    /**
     * It finds the current location of the mobile unit if available and permission granted
     * On create method loads the relevant cascadefile and model
     *
     * @param savedInstanceState
     */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "Called onCreate");
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.classification_activity);

        Toolbar toolbar = (Toolbar) findViewById(R.id.toolbar);
        setSupportActionBar(toolbar);

        //Loads opencv_java3 librady
        System.loadLibrary("opencv_java3");

        //Location Manager is found using the location service of current context.
        locationManager = (LocationManager) getSystemService(Context.LOCATION_SERVICE);

        //Checks wether or not the application has permissions to access the fine location defined in the Android manifest
        boolean permissionGranted = ActivityCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) == PackageManager.PERMISSION_GRANTED;

        //If permission is granted the location manager requests and updates the location
        if (permissionGranted) {
            locationManager.requestLocationUpdates(LocationManager.GPS_PROVIDER, 0, 0, this);
            Location location = locationManager.getLastKnownLocation(LocationManager.GPS_PROVIDER);
            if (location != null) {
                //In some cases the user has given explicit access to use the units location in this application
                //while the unit location is not actually available
                //This check prevents the application from crashing accessing a null object with get methods in empty location
                this.geoLocation = location.getLatitude() + ", " + location.getLongitude();
            }
        } else {
            //If permission is not granted, the application will request the permission to access the location
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.ACCESS_FINE_LOCATION}, 200);
        }

        //Accesses the cascade classifier and defining the cascade variable
        loadCascadeDetector();
        //Loading the machine learning model and defining the model variable
        loadModel();

        mOpenCvCameraView = findViewById(R.id.CameraView);
        mOpenCvCameraView.setCvCameraViewListener(this);

    }

    @Override
    public boolean onTouchEvent(MotionEvent event) {
        mOpenCvCameraView.focusOnTouch(event, Objects.requireNonNull(getSupportActionBar()).getHeight());
        return super.onTouchEvent(event);
    }


    @Override
    public void onPause() {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_3_0, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        mOpenCvCameraView.disableView();
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        mGray = new Mat();
        mRgba = new Mat();
    }

    @Override
    public void onCameraViewStopped() {
        mGray.release();
        mRgba.release();
    }


    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {

        mRgba = inputFrame.rgba();
        mGray = inputFrame.gray();
        if (cascadeDetector != null) {
            MatOfRect found = new MatOfRect();

            //Running cascade detector on image to detect bounding box of speed limits
            cascadeDetector.detectMultiScale(mGray, found, 1.4, 5, 0, new Size(24, 24), new Size());


            Rect[] rects = found.toArray();
            Mat traffic_sign = new Mat();

            //Looping through objects found by cascade classifier
            for (Rect rect : rects) {
                //x and y values of bounding box
                int xd1 = (int) rect.tl().x;
                int yd1 = (int) rect.tl().y;
                int xd2 = (int) rect.br().x;
                int yd2 = (int) rect.br().y;

                //Defining rect based on values
                Rect roi = new Rect(xd1, yd1, xd2 - xd1, yd2 - yd1);

                //Creating a copy of original image based on bounding box
                traffic_sign = mRgba.submat(roi).clone();

                //Displaying green rect on screen
                Imgproc.rectangle(mRgba, rect.tl(), rect.br(), OBJECT_RECT_COLOR, 3);

                //Resize image to fit model input size
                Size size = new Size(112, 112);
                Imgproc.resize(traffic_sign, traffic_sign, size, Imgproc.INTER_LANCZOS4);

                //Applying CLAHE on all channels by converting to lab format and applying CLAHE on instesity layer.
                Imgproc.cvtColor(traffic_sign, traffic_sign, Imgproc.COLOR_RGB2Lab);
                List<Mat> channels = new LinkedList();
                Core.split(traffic_sign, channels);
                CLAHE clahe = Imgproc.createCLAHE(2.0, new Size(8, 8));
                clahe.apply(channels.get(0), traffic_sign);
                Core.merge(channels, traffic_sign);

                //Converting back to regular rgb format
                Imgproc.cvtColor(traffic_sign, traffic_sign, Imgproc.COLOR_Lab2RGB);

                Bitmap bitmap = Bitmap.createBitmap(112, 112, Bitmap.Config.ARGB_8888);
                Utils.matToBitmap(traffic_sign, bitmap);

                //Defining custom vectors to not apply std and mead normalization
                float[] c1 = new float[]{.0f, .0f, .0f};
                float[] c2 = new float[]{1.0f, 1.0f, 1.0f};
                final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(bitmap,
                        c1, c2);

                IValue fl = IValue.from(inputTensor);

                Tensor outputTensor = module.forward(fl).toTensor();

                // getting tensor content as java array of floats
                final float[] scores = outputTensor.getDataAsFloatArray();


                // searching for the index with maximum score
                float maxScore = -Float.MAX_VALUE;
                int maxScoreIdx = -1;
                for (int j = 0; j < scores.length; j++) {
                    if (scores[j] > maxScore) {
                        maxScore = scores[j];
                        maxScoreIdx = j;
                        Log.d(String.valueOf(maxScoreIdx), String.valueOf(maxScore));
                    }
                }

                String className = TrafficClasses.TRAFFIC_CLASSES[maxScoreIdx];

                // showing className on UI
                TextView textView = findViewById(R.id.resultView);
                textView.setText(className);

                PredictionValidation.signPredictionList.add(className);
                PredictionValidation.signImageList.add(bitmap);
                //PredictionValidation.predictionMaxValue.add(maxScore);
                PredictionValidation.geoLocationList.add(geoLocation);
            }
        }

        return mRgba;
    }

    /**
     * Method for finding the correct path to the assets folder in the mobile unit
     * Context and assetName is used to find a given file in assets folder if it exists
     * If it does not already exists it is written as a new file in the same assets folder
     *
     * @param context
     * @param assetName
     * @return absolute path of file if found or created successfully
     * @throws IOException
     */
    private static String assetFilePath(Context context, String assetName) throws IOException {
        File file = new File(context.getFilesDir(), assetName);
        //If file exists and is not empty, returns the absolute path
        if (file.exists() && file.length() > 0) {
            Log.d("File found", file.getAbsolutePath());

            return file.getAbsolutePath();
        }
        //if not, writes the corresponding file and return the absolute path
        try (InputStream is = context.getAssets().open(assetName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        }
    }

    /**
     * Method for accessing and defining the cascade detector to be used for sign detection
     * If the method fails to find the cascade file it is set to null
     */
    private void loadCascadeDetector(){
        try {
            mCascadeFile = new File(assetFilePath(this, "haarcascade.xml"));
            cascadeDetector = new CascadeClassifier(mCascadeFile.getAbsolutePath());
            if (cascadeDetector.empty()) {
                Log.e(TAG, "Failed to load cascade classifier");
                cascadeDetector = null;
            } else
                Log.i(TAG, "Loaded cascade classifier from " + mCascadeFile.getAbsolutePath());
        } catch (IOException e) {
            cascadeDetector = null;
            e.printStackTrace();
        }
    }

    /**
     * Method for accesses the model using pytorch Module.load method with the file path to the assets folder
     * If the model is not loaded successfully an exception is logged
     */
    private void loadModel(){
        try {
            //Change model by changing asset name: model9classess.pt or model43classes.pt
            module = Module.load(assetFilePath(this, "model9classes.pt"));
        } catch (IOException e) {
            Log.e("ERROR", "Error reading assets", e);
            finish();
        }
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        getMenuInflater().inflate(R.menu.main_menu, menu);
        return super.onCreateOptionsMenu(menu);
    }

    @Override
    public boolean onOptionsItemSelected(@NonNull MenuItem item) {
        startActivity(new Intent(this, PredictionValidation.class));
        return super.onOptionsItemSelected(item);
    }

    /**
     * The location listener updates the location when applicable
     * The class variable geoLocation is updated so it can be used whenever an sign is detected
     *
     * @param location
     */
    @Override
    public void onLocationChanged(Location location) {
        this.geoLocation = location.getLatitude() + ", " + location.getLongitude();
    }

    @Override
    public void onProviderDisabled(String provider) {
        Log.d("Latitude", "disable");
    }

    @Override
    public void onProviderEnabled(String provider) {
        Log.d("Latitude","enable");
    }
}

