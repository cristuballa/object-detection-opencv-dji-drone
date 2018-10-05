package com.dji.videostreamdecodingsample;

import android.app.Activity;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.ImageFormat;

import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.PointF;
import android.graphics.Rect;
import android.graphics.RectF;
import android.graphics.SurfaceTexture;
import android.graphics.YuvImage;
import android.media.FaceDetector;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.os.Looper;
import android.os.Message;
import android.util.Log;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.TextureView;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;

import com.dji.videostreamdecodingsample.media.DJIVideoStreamDecoder;

import com.dji.videostreamdecodingsample.media.NativeHelper;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import dji.common.camera.SettingsDefinitions;
import dji.common.error.DJIError;
import dji.common.util.CommonCallbacks;
import dji.log.DJILog;
import dji.thirdparty.afinal.core.AsyncTask;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;

import dji.common.product.Model;
import dji.sdk.base.BaseProduct;
import dji.sdk.camera.Camera;
import dji.sdk.camera.VideoFeeder;
import dji.sdk.codec.DJICodecManager;
//import dji.thirdparty.afinal.utils.Utils;

import java.nio.ByteBuffer;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.Stack;

import static org.opencv.core.Core.flip;


import dji.common.error.DJIError;
import dji.common.flightcontroller.FlightControllerState;
import dji.common.flightcontroller.ObstacleDetectionSector;
import dji.common.flightcontroller.VisionDetectionState;
import dji.common.flightcontroller.virtualstick.FlightControlData;
import dji.common.flightcontroller.virtualstick.FlightCoordinateSystem;
import dji.common.flightcontroller.virtualstick.RollPitchControlMode;
import dji.common.flightcontroller.virtualstick.VerticalControlMode;
import dji.common.flightcontroller.virtualstick.YawControlMode;
import dji.common.util.CommonCallbacks;
import dji.sdk.flightcontroller.FlightAssistant;
import dji.sdk.flightcontroller.FlightController;
import dji.sdk.products.Aircraft;


public class MainActivity extends Activity implements DJICodecManager.YuvDataCallback {
    private static final String TAG = MainActivity.class.getSimpleName();

    static private Aircraft aircraft = new Aircraft(null);
    static private FlightController flightController = aircraft.getFlightController();
    static private FlightAssistant flightAssistant = flightController.getFlightAssistant();
    static private Stack<Float> flightMoves = new Stack<>();
    double focalLength;
    private static final int MSG_WHAT_SHOW_TOAST = 0;
    private static final int MSG_WHAT_UPDATE_TITLE = 1;
    private SurfaceHolder.Callback surfaceCallback;
    private enum DemoType { USE_TEXTURE_VIEW, USE_SURFACE_VIEW, USE_SURFACE_VIEW_DEMO_DECODER}
    private static DemoType demoType = DemoType.USE_SURFACE_VIEW_DEMO_DECODER;
    private VideoFeeder.VideoFeed standardVideoFeeder;
    private Button btnStartFaceRecognition;
    Mat bwIMG, hsvIMG, lrrIMG, urrIMG, dsIMG, usIMG, cIMG, hovIMG;
    MatOfPoint2f approxCurve;
    double distance;

    byte[] m_yuvFrame;
    int m_yuvWidth;
    int m_yuvHeight;
    int m_iShow;
    int m_iFrameNumber;

    float x_coordinate;
    float y_coordinate;
    boolean object_detected;

    protected Bitmap mBitMapImage;
    protected byte[] mBytes;
    Paint mPaint = new Paint();
    PointF m_facesMidPoint = null;
    float m_facesConfidence = 0.0f;
    float m_facesDistance = 0.0f;
    int mframeWidth = 0;
    int mframeHeight = 0;
    int m_iShowFindFace = 0;
    int m_TryToFindFaces = 0;
    long timePrevFrame = 0;
    long timeDelta;
    long timeNow;

    CanvasThread canvasThread;

    int threshold;

    protected VideoFeeder.VideoDataCallback mReceivedVideoDataCallBack = null;

    private TextView titleTv;
    public Handler mainHandler = new Handler(Looper.getMainLooper()) {
        @Override
        public void handleMessage(Message msg) {
            switch (msg.what) {
                case MSG_WHAT_SHOW_TOAST:
                    Toast.makeText(getApplicationContext(), (String) msg.obj, Toast.LENGTH_SHORT).show();
                    break;
                case MSG_WHAT_UPDATE_TITLE:
                    if (titleTv != null) {
                        titleTv.setText((String) msg.obj);
                    }
                    break;
                default:
                    break;
            }
        }
    };

    private TextureView videostreamPreviewTtView;
    private SurfaceView videostreamPreviewSf;
    private SurfaceHolder videostreamPreviewSh;
    private Camera mCamera;
    private DJICodecManager mCodecManager;
    private TextView savePath;
    private Button screenShot;
    private StringBuilder stringBuilder;
    private int videoViewWidth;
    private int videoViewHeight;
    private int count;

    @Override
    protected void onResume() {
        super.onResume();
        initSurfaceOrTextureView();
        notifyStatusChange();
        DJIVideoStreamDecoder.getInstance().resume();
    }

    private void initSurfaceOrTextureView(){
        switch (demoType) {
            case USE_SURFACE_VIEW:
                initPreviewerSurfaceView();
                break;
            case USE_SURFACE_VIEW_DEMO_DECODER:
                /**
                 * we also need init the textureView because the pre-transcoded video steam will display in the textureView
                 */
                initPreviewerTextureView();

                /**
                 * we use standardVideoFeeder to pass the transcoded video data to DJIVideoStreamDecoder, and then display it
                 * on surfaceView
                 */
                initPreviewerSurfaceView();
                break;
            case USE_TEXTURE_VIEW:
                initPreviewerTextureView();
                break;
        }
    }

    @Override
    protected void onPause() {
        if (mCamera != null) {
            if (VideoFeeder.getInstance().getPrimaryVideoFeed() != null) {
                VideoFeeder.getInstance().getPrimaryVideoFeed().setCallback(null);
            }
            if (standardVideoFeeder != null) {
                standardVideoFeeder.setCallback(null);
            }
        }

        boolean retry = true;
        if (canvasThread != null) {
            canvasThread.setRunning(false);
            while (retry) {
                try {
                    canvasThread.join();
                    retry = false;
                } catch (InterruptedException e) {

                }
            }
        }
        DJIVideoStreamDecoder.getInstance().stop();
        super.onPause();
        finish();
    }

    @Override
    protected void onDestroy() {
        if (mCodecManager != null) {
            mCodecManager.cleanSurface();
            mCodecManager.destroyCodec();
        }
        super.onDestroy();
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {

        if (OpenCVLoader.initDebug())  {
            Toast.makeText(getApplicationContext(),"OpenCv loaded successfully",Toast.LENGTH_SHORT).show();
        }
        else {
            Toast.makeText(getApplicationContext(),"OpenCV could not be loaded",Toast.LENGTH_SHORT).show();
        }
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        initUi();
        bwIMG = new Mat();
        dsIMG = new Mat();
        hsvIMG = new Mat();
        lrrIMG = new Mat();
        urrIMG = new Mat();
        usIMG = new Mat();
        cIMG = new Mat();
        hovIMG = new Mat();
        approxCurve = new MatOfPoint2f();
    }

    private void showToast(String s) {
        mainHandler.sendMessage(
                mainHandler.obtainMessage(MSG_WHAT_SHOW_TOAST, s)
        );
    }


    private void updateTitle(String s) {
        mainHandler.sendMessage(
                mainHandler.obtainMessage(MSG_WHAT_UPDATE_TITLE, s)
        );
    }


    private void initUi() {
        btnStartFaceRecognition =findViewById(R.id.btnFrames);
        savePath = findViewById(R.id.activity_main_save_path);
        screenShot =findViewById(R.id.activity_main_screen_shot);
        screenShot.setSelected(false);

        titleTv =findViewById(R.id.title_tv);
        videostreamPreviewTtView =findViewById(R.id.livestream_preview_ttv);
        videostreamPreviewSf = findViewById(R.id.livestream_preview_sf);
        videostreamPreviewSf.setClickable(true);
        videostreamPreviewSf.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                float rate = VideoFeeder.getInstance().getTranscodingDataRate();
                showToast("current rate:" + rate + "Mbps");
                if (rate < 10) {
                    VideoFeeder.getInstance().setTranscodingDataRate(10.0f);
                    showToast("set rate to 10Mbps");
                } else {
                    VideoFeeder.getInstance().setTranscodingDataRate(3.0f);
                    showToast("set rate to 3Mbps");
                }
            }
        });
        updateUIVisibility();
    }


    private void updateUIVisibility(){
        switch (demoType) {
            case USE_SURFACE_VIEW:
                videostreamPreviewSf.setVisibility(View.VISIBLE);
                videostreamPreviewTtView.setVisibility(View.GONE);
                break;

            case USE_SURFACE_VIEW_DEMO_DECODER:
                /**
                 * we need display two video stream at the same time, so we need let them to be visible.
                 */
                videostreamPreviewSf.setVisibility(View.VISIBLE);
                videostreamPreviewTtView.setVisibility(View.VISIBLE);
                break;

            case USE_TEXTURE_VIEW:
                videostreamPreviewSf.setVisibility(View.GONE);
                videostreamPreviewTtView.setVisibility(View.VISIBLE);
                break;
        }
    }

    private long lastupdate;
    private void notifyStatusChange() {

        final BaseProduct product = VideoDecodingApplication.getProductInstance();

        Log.d(TAG, "notifyStatusChange: " + (product == null ? "Disconnect" : (product.getModel() == null ? "null model" : product.getModel().name())));
        if (product != null && product.isConnected() && product.getModel() != null) {
            updateTitle(product.getModel().name() + " Connected " + demoType.name());
        } else {
            updateTitle("Disconnected");
        }

        // The callback for receiving the raw H264 video data for camera live view
        mReceivedVideoDataCallBack = new VideoFeeder.VideoDataCallback() {

            @Override
            public void onReceive(byte[] videoBuffer, int size) {
                if (System.currentTimeMillis() - lastupdate > 1000) {
                    Log.d(TAG, "camera recv video data size: " + size);
                    lastupdate = System.currentTimeMillis();
                }
                switch (demoType) {
                    case USE_SURFACE_VIEW:
                        if (mCodecManager != null) {
                            mCodecManager.sendDataToDecoder(videoBuffer, size);
                        }
                        break;
                    case USE_SURFACE_VIEW_DEMO_DECODER:
                        /**
                         we use standardVideoFeeder to pass the transcoded video data to DJIVideoStreamDecoder, and then display it
                         * on surfaceView
                         */
                        DJIVideoStreamDecoder.getInstance().parse(videoBuffer, size);
                        break;

                    case USE_TEXTURE_VIEW:
                        if (mCodecManager != null) {
                            mCodecManager.sendDataToDecoder(videoBuffer, size);
                        }
                        break;
                }
            }
        };



        if (null == product || !product.isConnected()) {
            mCamera = null;
            showToast("Disconnected");
        } else {
            if (!product.getModel().equals(Model.UNKNOWN_AIRCRAFT)) {
                mCamera = product.getCamera();
                mCamera.setMode(SettingsDefinitions.CameraMode.SHOOT_PHOTO, new CommonCallbacks.CompletionCallback() {
                    @Override
                    public void onResult(DJIError djiError) {
                        if (djiError != null) {
                            showToast("can't change mode of camera, error:"+djiError.getDescription());
                        }
                    }
                });

                if (demoType == DemoType.USE_SURFACE_VIEW_DEMO_DECODER) {
                    if (VideoFeeder.getInstance() != null) {
                        standardVideoFeeder = VideoFeeder.getInstance().provideTranscodedVideoFeed();
                        standardVideoFeeder.setCallback(mReceivedVideoDataCallBack);
                    }
                } else {
                    if (VideoFeeder.getInstance().getPrimaryVideoFeed() != null) {
                        VideoFeeder.getInstance().getPrimaryVideoFeed().setCallback(mReceivedVideoDataCallBack);
                    }
                }
            }
        }
    }

    private void takeOff() {
        // Take off action
        flightController.startTakeoff(null);
    }

    private void Pitch2(int distance) {

        flightController.setRollPitchControlMode(RollPitchControlMode.VELOCITY);
        flightController.setRollPitchCoordinateSystem(FlightCoordinateSystem.BODY);
        flightController.sendVirtualStickFlightControlData(new FlightControlData(15f, 0, 0, 0), null);
        
    }

    private void land() {
        // Land and on result give control back to Executor
        flightController.startLanding(new CommonCallbacks.CompletionCallback() {
            @Override
            public void onResult(DJIError djiError) {
            }
        });
    }

    /**
     * Init a fake texture view to for the codec manager, so that the video raw data can be received
     * by the camera
     */
    private void initPreviewerTextureView() {
        videostreamPreviewTtView.setSurfaceTextureListener(new TextureView.SurfaceTextureListener() {
            @Override
            public void onSurfaceTextureAvailable(SurfaceTexture surface, int width, int height) {
                Log.d(TAG, "real onSurfaceTextureAvailable");
                videoViewWidth = width;
                videoViewHeight = height;
                Log.d(TAG, "real onSurfaceTextureAvailable: width " + videoViewWidth + " height " + videoViewHeight);
                if (mCodecManager == null) {
                    mCodecManager = new DJICodecManager(getApplicationContext(), surface, width, height);
                }
            }

            @Override
            public void onSurfaceTextureSizeChanged(SurfaceTexture surface, int width, int height) {
                videoViewWidth = width;
                videoViewHeight = height;
                Log.d(TAG, "real onSurfaceTextureAvailable2: width " + videoViewWidth + " height " + videoViewHeight);

                if (object_detected==true){
                   //Pitch
                    showToast("Object Detected: " );
                }


            }

            @Override
            public boolean onSurfaceTextureDestroyed(SurfaceTexture surface) {
                if (mCodecManager != null) {
                    mCodecManager.cleanSurface();
                }
                return false;
            }

            @Override
            public void onSurfaceTextureUpdated(SurfaceTexture surface) {
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        // displayPath(path);
                        //DetectCircle();
                    }
                });
            }
        });
    }

    /**
     * Init a surface view for the DJIVideoStreamDecoder
     */
    private void initPreviewerSurfaceView() {
        videostreamPreviewSh = videostreamPreviewSf.getHolder();
        surfaceCallback = new SurfaceHolder.Callback() {
            @Override
            public void surfaceCreated(SurfaceHolder holder) {
                Log.d(TAG, "real onSurfaceTextureAvailable");
                videoViewWidth = videostreamPreviewSf.getWidth();
                videoViewHeight = videostreamPreviewSf.getHeight();
                Log.d(TAG, "real onSurfaceTextureAvailable3: width " + videoViewWidth + " height " + videoViewHeight);
                switch (demoType) {
                    case USE_SURFACE_VIEW:
                        if (mCodecManager == null) {
                            mCodecManager = new DJICodecManager(getApplicationContext(), holder, videoViewWidth,videoViewHeight);
                        }
                        break;
                    case USE_SURFACE_VIEW_DEMO_DECODER:
                        // This demo might not work well on P3C and OSMO.
                        NativeHelper.getInstance().init();
                        DJIVideoStreamDecoder.getInstance().init(getApplicationContext(), holder.getSurface());
                        DJIVideoStreamDecoder.getInstance().resume();
                        break;
                }
            }

            @Override
            public void surfaceChanged(SurfaceHolder holder, int format, int width, int height) {
                videoViewWidth = width;
                videoViewHeight = height;
                Log.d(TAG, "real onSurfaceTextureAvailable4: width " + videoViewWidth + " height " + videoViewHeight);
                switch (demoType) {
                    case USE_SURFACE_VIEW:
                        //mCodecManager.onSurfaceSizeChanged(videoViewWidth, videoViewHeight, 0);
                        break;
                    case USE_SURFACE_VIEW_DEMO_DECODER:
                     //   showToast("DJIVideoStreamDecoder.getInstance().changeSurface(holder.getSurface());");
                       // tryDrawing(holder);
                        DJIVideoStreamDecoder.getInstance().changeSurface(holder.getSurface());
                        break;
                }
            }


            @Override
            public void surfaceDestroyed(SurfaceHolder holder) {
                showToast("Surfaced Destroyed");
                switch (demoType) {
                    case USE_SURFACE_VIEW:
                        if (mCodecManager != null) {
                            mCodecManager.cleanSurface();
                            mCodecManager.destroyCodec();
                            mCodecManager = null;
                        }
                        break;
                    case USE_SURFACE_VIEW_DEMO_DECODER:
                        DJIVideoStreamDecoder.getInstance().stop();
                        NativeHelper.getInstance().release();
                        break;
                }
                boolean retry = true;
                if (canvasThread != null) {
                    canvasThread.setRunning(false);
                    while (retry) {
                        try {
                            canvasThread.join();
                            retry = false;
                        } catch (InterruptedException e) {

                        }
                    }
                }

            }
        };

        videostreamPreviewSh.addCallback(surfaceCallback);
    }



    // public void onYuvDataReceived(final ByteBuffer yuvFrame, int dataSize, final int width, final int height) {
    @Override
    public void onYuvDataReceived(final ByteBuffer yuvFrame, int dataSize, final int width, final int height) {

        //DJILog.d(TAG, "onYuvDataReceived " + dataSize);
       // if (count++ % 30 == 0 && yuvFrame != null) {
       // if (m_iShow == 1) {
       //     m_iShow = 0;
            final byte[] bytes = new byte[dataSize];
            yuvFrame.get(bytes);
            //DJILog.d(TAG, "onYuvDataReceived2 " + dataSize);
           // m_yuvFrame = java.util.Arrays.copyOf(bytes, bytes.length);
           m_yuvFrame= bytes;
            //showToast("Frame length "  + bytes.length);
            m_yuvWidth = width;
            m_yuvHeight = height;
            //AsyncTask.execute(new Runnable() {
            //    @Override
            //    public void run() {

                   //saveYuvDataToJPEG(bytes, width, height);
                    //showToast("createBitmapFromYuvFrame");
                    //createBitmapFromYuvFrame(bytes,width,height);
                   // DetectShape(;
                    // YUV data received one frame at a time.

                 //   saveYuvDataToJPEG(bytes, width, height);
           //     }
          //  });
        //}   else {
        //m_iShow++;
 //   }



    }

    private void saveYuvDataToJPEG(byte[] yuvFrame, int width, int height){

        if (yuvFrame.length < width * height) {
            //DJILog.d(TAG, "yuvFrame size is too small " + yuvFrame.length);
            return;

        }

        m_yuvFrame = java.util.Arrays.copyOf(yuvFrame, yuvFrame.length);
        showToast("Frame length "  + yuvFrame.length);
        m_yuvWidth = width;
        m_yuvHeight = height;

        /*
        byte[] y = new byte[width * height];
        byte[] u = new byte[width * height / 4];
        byte[] v = new byte[width * height / 4];
        byte[] nu = new byte[width * height / 4]; //
        byte[] nv = new byte[width * height / 4];

        System.arraycopy(yuvFrame, 0, y, 0, y.length);
        for (int i = 0; i < u.length; i++) {
            v[i] = yuvFrame[y.length + 2 * i];
            u[i] = yuvFrame[y.length + 2 * i + 1];
        }
        int uvWidth = width / 2;
        int uvHeight = height / 2;
        for (int j = 0; j < uvWidth / 2; j++) {
            for (int i = 0; i < uvHeight / 2; i++) {
                byte uSample1 = u[i * uvWidth + j];
                byte uSample2 = u[i * uvWidth + j + uvWidth / 2];
                byte vSample1 = v[(i + uvHeight / 2) * uvWidth + j];
                byte vSample2 = v[(i + uvHeight / 2) * uvWidth + j + uvWidth / 2];
                nu[2 * (i * uvWidth + j)] = uSample1;
                nu[2 * (i * uvWidth + j) + 1] = uSample1;
                nu[2 * (i * uvWidth + j) + uvWidth] = uSample2;
                nu[2 * (i * uvWidth + j) + 1 + uvWidth] = uSample2;
                nv[2 * (i * uvWidth + j)] = vSample1;
                nv[2 * (i * uvWidth + j) + 1] = vSample1;
                nv[2 * (i * uvWidth + j) + uvWidth] = vSample2;
                nv[2 * (i * uvWidth + j) + 1 + uvWidth] = vSample2;
            }

        }


        //nv21test
        byte[] bytes = new byte[yuvFrame.length];
        System.arraycopy(y, 0, bytes, 0, y.length);

        for (int i = 0; i < u.length; i++) {
            bytes[y.length + (i * 2)] = nv[i];
            bytes[y.length + (i * 2) + 1] = nu[i];
        }                   if (m_iShow == 1) {
            m_iShow = 0;

            Log.e(TAG, "Receving frame" + m_iFrameNumber++);
            // Set m_yuvFrame which is then drawn in another thread.
            m_yuvFrame = java.util.Arrays.copyOf(yuvFrame, yuvFrame.length);
            showToast("Frame length "  + yuvFrame.length);
            m_yuvWidth = width;
            m_yuvHeight = height;

        } else {
            m_iShow++;
        }
*/
    }

 public void DetectCircle(Mat mat)
 {

     Mat  upper_red_hue_range= new Mat();
   //  Mat gray = new Mat();
     Mat mHSV=new Mat();
     /*
     Bitmap bmp32 = bitmap.copy(Bitmap.Config.ARGB_8888, true);
     Utils.bitmapToMat(bmp32, gray);
*/  // showToast("Imgproc.cvtColor");

     Imgproc.cvtColor(mat, mHSV, Imgproc.COLOR_RGB2HSV, 3); //3 is HSV Channel
     Core.inRange(mHSV, new Scalar(160, 100, 100), new Scalar(179, 255, 255), upper_red_hue_range);

     Imgproc.erode(upper_red_hue_range, upper_red_hue_range, Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(5,5)));
     Imgproc.GaussianBlur(upper_red_hue_range, upper_red_hue_range, new Size(9, 9), 2, 2);

     Mat circles = new Mat();
     Imgproc.HoughCircles(upper_red_hue_range, circles,  Imgproc.CV_HOUGH_GRADIENT, 1, 100, 70,20, 30, 0);
     int numberOfCircles = (circles.rows() == 0) ? 0 : circles.cols();
     if (numberOfCircles> 0) {
         Log.d(TAG, "Number of circle" + numberOfCircles);
         showToast("Number of circle" + numberOfCircles);
     }

     showToast("Number of circle: " + numberOfCircles);
 }


    // http://www.41post.com/3470/programming/android-retrieving-the-camera-preview-as-a-pixel-array
    void decodeYUV420SP(int[] rgb, byte[] yuv420sp, int width, int height) {
        final int frameSize = width * height;
        for (int j = 0, yp = 0; j < height; j++) {  int uvp = frameSize + (j >> 1) * width, u = 0, v = 0;
            for (int i = 0; i < width; i++, yp++) {
                int y = (0xff & ((int) yuv420sp[yp])) - 16;
                if (y < 0)
                    y = 0;
                if ((i & 1) == 0) {
                    v = (0xff & yuv420sp[uvp++]) - 128;
                    u = (0xff & yuv420sp[uvp++]) - 128;
                }

                int y1192 = 1192 * y;
                int r = (y1192 + 1634 * v);
                int g = (y1192 - 833 * v - 400 * u);
                int b = (y1192 + 2066 * u);

                if (r < 0)                  r = 0;               else if (r > 262143)
                    r = 262143;
                if (g < 0)                  g = 0;               else if (g > 262143)
                    g = 262143;
                if (b < 0)                  b = 0;               else if (b > 262143)
                    b = 262143;

                rgb[yp] = 0xff000000 | ((r << 6) & 0xff0000) | ((g >> 2) & 0xff00) | ((b >> 10) & 0xff);
            }
        }
    }


    public  Bitmap getBitmapImageFromYUV(byte[] data, int width, int height) {

        YuvImage yuvimage = new YuvImage(data, ImageFormat.NV21, width, height, null);
        ByteArrayOutputStream baos = new ByteArrayOutputStream();

        yuvimage.compressToJpeg(new Rect(0, 0, width, height), 80, baos);
        byte[] jdata = baos.toByteArray();
        BitmapFactory.Options bitmapFatoryOptions = new BitmapFactory.Options();
        bitmapFatoryOptions.inPreferredConfig = Bitmap.Config.RGB_565;
        Bitmap bmp = BitmapFactory.decodeByteArray(jdata, 0, jdata.length, bitmapFatoryOptions);
        return bmp;

    }

  //  public class DetectRectangle implements Runnable {

   //      Bitmap _bitmap;
   //     DetectRectangle(Bitmap bitmap ){
   //          this._bitmap= bitmap;
   //      }

    //     public void run() {
   //          synchronized (this) {
   //              DetectShape(this._bitmap);
   //          }
    //     }
     //}

    /**
     * Save the buffered data into a JPG image file
     */

    private void screenShot(byte[] buf, String shotDir, int width, int height) {
        File dir = new File(shotDir);
        if (!dir.exists() || !dir.isDirectory()) {
            dir.mkdirs();
        }
        YuvImage yuvImage = new YuvImage(buf,
                ImageFormat.NV21,
                width,
                height,
                null);
        OutputStream outputFile;
        final String path = dir + "/ScreenShot_" + System.currentTimeMillis() + ".jpg";
        try {
            outputFile = new FileOutputStream(new File(path));
        } catch (FileNotFoundException e) {
            Log.e(TAG, "test screenShot: new bitmap output file error: " + e);
            return;
        }
        if (outputFile != null) {
            yuvImage.compressToJpeg(new Rect(0,
                    0,
                    width,
                    height), 100, outputFile);
        }
        try {
            outputFile.close();
        } catch (IOException e) {
            Log.e(TAG, "test screenShot: compress yuv image error: " + e);
            e.printStackTrace();
        }
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
               // displayPath(path);
            }
        });
    }


    public void onClick(View v) {

        if (v.getId() == R.id.activity_main_screen_shot) {
            handleYUVClick();
        } else {
            DemoType newDemoType = null;
            if (v.getId() == R.id.activity_main_screen_texture) {
                newDemoType = DemoType.USE_TEXTURE_VIEW;
            } else if (v.getId() == R.id.activity_main_screen_surface) {
                newDemoType = DemoType.USE_SURFACE_VIEW;
            } else if (v.getId() == R.id.activity_main_screen_surface_with_own_decoder) {
                newDemoType = DemoType.USE_SURFACE_VIEW_DEMO_DECODER;
            }

            if (newDemoType != null && newDemoType != demoType) {
                // Although finish will trigger onDestroy() is called, but it is not called before OnCreate of new activity.
                if (mCodecManager != null) {
                    mCodecManager.cleanSurface();
                    mCodecManager.destroyCodec();
                    mCodecManager = null;
                }
                demoType = newDemoType;
                finish();
                overridePendingTransition(0, 0);
                startActivity(getIntent());
                overridePendingTransition(0, 0);
            }
        }
    }

    private void handleYUVClick() {
        land();
        /*
        if (screenShot.isSelected()) {
            screenShot.setText("YUV Screen Shot");
            screenShot.setSelected(false);

            switch (demoType) {
                case USE_SURFACE_VIEW:
                case USE_TEXTURE_VIEW:
                    mCodecManager.enabledYuvData(false);
                    mCodecManager.setYuvDataCallback(null);
                    // ToDo:
                    break;
                case USE_SURFACE_VIEW_DEMO_DECODER:

                    DJIVideoStreamDecoder.getInstance().changeSurface(videostreamPreviewSh.getSurface());
                    DJIVideoStreamDecoder.getInstance().setYuvDataListener(null);
                    break;

            }
            savePath.setText("");
            savePath.setVisibility(View.INVISIBLE);
            stringBuilder = null;
        } else {
            screenShot.setText("Live Stream");
            screenShot.setSelected(true);

            switch (demoType) {
                case USE_TEXTURE_VIEW:
                case USE_SURFACE_VIEW:
                    mCodecManager.enabledYuvData(true);
                    mCodecManager.setYuvDataCallback(this);
                    break;
                case USE_SURFACE_VIEW_DEMO_DECODER:
                    DJIVideoStreamDecoder.getInstance().changeSurface(null);
                    DJIVideoStreamDecoder.getInstance().setYuvDataListener(MainActivity.this);
                    break;
            }
            savePath.setText("");
           // savePath.setVisibility(View.VISIBLE);
        }

        */


    }

    private void displayPath(String path) {
        if (stringBuilder == null) {
            stringBuilder = new StringBuilder();
        }

        path = path + "\n";
        stringBuilder.append(path);
        savePath.setText(stringBuilder.toString());
    }


///=============================Codes added============================//

    private void useLiveStream() {
        boolean retry = true;
        if (canvasThread != null) {
            canvasThread.setRunning(false);
            while (retry) {
                try {
                    canvasThread.join();
                    retry = false;
                } catch (InterruptedException e) {

                }
            }
        }

        DJIVideoStreamDecoder.getInstance().changeSurface(videostreamPreviewSh.getSurface());
    }

    private void useFrames() {
        try {
            DJIVideoStreamDecoder.getInstance().changeSurface(null);
            DJIVideoStreamDecoder.getInstance().setYuvDataListener(MainActivity.this);


            if (canvasThread == null) {
             //   showToast(" canvasThread not null");
                canvasThread = new CanvasThread(videostreamPreviewSh);
            }
           // showToast("Trigged surface destroyed" );
            canvasThread.setRunning(true);
            canvasThread.start();

        } catch (Exception e) {
            showToast("UseFrames"+ e.getMessage());
        }


    }


    public void btnLiveStream(View v) {
        useLiveStream();
    }


    private Bitmap createBitmapFromYuvFrame( byte[] yuvFrame, int yuv_Width, int yuv_Height) {

        Mat mat_newBuf = new Mat(yuv_Height + yuv_Height / 2, yuv_Width, CvType.CV_8UC1);
        mat_newBuf.put(0, 0, yuvFrame);

        Mat matBRGA = new Mat(DJIVideoStreamDecoder.getInstance().height, DJIVideoStreamDecoder.getInstance().width, CvType.CV_8UC4);
        Imgproc.cvtColor(mat_newBuf, matBRGA, Imgproc.COLOR_YUV420sp2BGRA);

        matBRGA= DetectRect(matBRGA);
        // convert to bitmap:
        Bitmap bm_Format_565 = Bitmap.createBitmap(matBRGA.cols(), matBRGA.rows(), Bitmap.Config.RGB_565);
      //  Utils.matToBitmap(matBRGA, bm_Format_565);
        Utils.matToBitmap(matBRGA, bm_Format_565);

        mframeWidth = bm_Format_565.getWidth();
        mframeHeight = bm_Format_565.getHeight();

        mat_newBuf.release();
        matBRGA.release();
        //showToast("Bitmap Return 4");
        return bm_Format_565;

    }


    private void updateCanvas(final Canvas canvas, Bitmap bm, int canvasWidth, int canvasHeight) {
        canvas.drawBitmap(bm, null, new RectF(0, 0, canvasWidth, canvasHeight), mPaint);
    }


    public Mat DetectRect(Mat rgba ) {

        Mat mHSV = new Mat();
        Mat  upper_red_hue_range= new Mat();
        object_detected=false;
        Imgproc.cvtColor(rgba, mHSV, Imgproc.COLOR_RGB2HSV, 3); //3 is HSV Channel
        //red
        Core.inRange(mHSV, new Scalar(160, 100, 100), new Scalar(179, 255, 255), upper_red_hue_range);
        //Blue
        //Core.inRange(mHSV, new Scalar(100,150,0), new Scalar(140,255,255), upper_red_hue_range);
        //Orange
        // Core.inRange(mHSV, new Scalar(5,50,50), new Scalar(15,255,255), upper_red_hue_range);
        Imgproc.pyrDown(upper_red_hue_range, dsIMG, new Size(upper_red_hue_range.cols() / 2, upper_red_hue_range.rows() / 2));
        Imgproc.pyrUp(dsIMG, usIMG, upper_red_hue_range.size());
        Imgproc.Canny(usIMG, bwIMG, 0, threshold);
        Imgproc.dilate(bwIMG, bwIMG, new Mat(), new Point(-1, 1), 1);

        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
        cIMG = bwIMG.clone();
        Imgproc.findContours(cIMG, contours, hovIMG, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        for (MatOfPoint cnt : contours) {

            MatOfPoint2f curve = new MatOfPoint2f(cnt.toArray());
            Imgproc.approxPolyDP(curve, approxCurve, 0.02 * Imgproc.arcLength(curve, true), true);
            int numberVertices = (int) approxCurve.total();
            double contourArea = Imgproc.contourArea(cnt);
            if (Math.abs(contourArea) < 100) {
                continue;
            }

            //Rectangle detected
            if (numberVertices >= 4 && numberVertices <= 6) {
                List<Double> cos = new ArrayList<>();
                for (int j = 2; j < numberVertices + 1; j++) {
                    cos.add(angle(approxCurve.toArray()[j % numberVertices], approxCurve.toArray()[j - 2], approxCurve.toArray()[j - 1]));
                }

                Collections.sort(cos);

                double mincos = cos.get(0);
                double maxcos = cos.get(cos.size() - 1);

                if (numberVertices == 4 && mincos >= -0.1 && maxcos <= 0.3) {
                    setLabel(rgba, "X", cnt);
                    object_detected=true;
                }

            }

        }
        return rgba;

    }

    private  double angle(Point pt1, Point pt2, Point pt0) {
        double dx1 = pt1.x - pt0.x;
        double dy1 = pt1.y - pt0.y;
        double dx2 = pt2.x - pt0.x;
        double dy2 = pt2.y - pt0.y;
        return (dx1 * dx2 + dy1 * dy2) / Math.sqrt((dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2) + 1e-10);
    }


    private void setLabel(Mat im, String label, MatOfPoint contour) {
        int fontface = Core.FONT_HERSHEY_SIMPLEX;
        double scale = 3;//0.4;
        int thickness = 10;//1;
        int[] baseline = new int[1];

        DecimalFormat df2 = new DecimalFormat(".##");
        Size text = Imgproc.getTextSize(label, fontface, scale, thickness, baseline);
        org.opencv.core.Rect r = Imgproc.boundingRect(contour);
        Point pt = new Point(r.x + ((r.width - text.width) / 2),r.y + ((r.height + text.height) / 2));
        Point pt2 = new Point(r.x ,r.y -20);
        Point pt3 = new Point(r.x ,r.y -90);
        // F=(PxD)/W
        // D=WF/P

        x_coordinate=r.x;
        y_coordinate=r.y;

        double W=14.7;
        double P=256;
        double D=50;
        double F= ((P*D)/W);

        String label2= "Distance= " + df2.format((F*W)/(r.width)) + " cm";
       //String label2= "Distance= " + df2.format((0.180446f*1080.6557f)/(r.width)) + " ft. pixel:" + r.width  ;
      //String label2= "Distance= " +  df2.format(1739.00f/(r.width*r.height)) +"meter" ;

        String label3= "P(" + Double.toString(r.x + ((r.width - text.width) / 2)) + ","+ Double.toString(r.y + ((r.height + text.height) / 2))+ ")";

        Imgproc.putText(im, label3, pt2, fontface, 2, new Scalar(0, 255, 0, 255), 5);
        Imgproc.putText(im, label2, pt3, fontface, 2, new Scalar(0, 255, 0, 255), 5);
      //  Imgproc.putText(im, label, pt, fontface, scale, new Scalar(0, 255, 0, 255), thickness);
        Imgproc.rectangle(im, new Point(r.x, r.y), new Point(r.x + r.width, r.y + r.height), new Scalar(0, 255, 0, 255), 10);
       // showToast("Object Detected: " + object_detected);
    }
    public void btnFrames(View v) {
        //useLiveStream();
        useFrames();
        // btnStartFaceRecognition.setEnabled(false);
    }


    public void draw(Canvas canvas) {
        if (m_yuvFrame != null) {
            if (m_yuvFrame.length > 0) {
                if (videostreamPreviewSh != null) {
                    showFrame(canvas, videostreamPreviewSh, m_yuvFrame, m_yuvWidth, m_yuvHeight);
                }
            }
        }
    }


    private void showFrame(Canvas canvas, SurfaceHolder holder, byte[] yuvFrame, int yuv_width, int yuv_height) {
        if (holder != null) {
            //Canvas canvas = holder.lockCanvas();
            // Convert YUV to bitmap.
            if (mBitMapImage != null) {
                mBitMapImage.recycle();
            }
            //  showToast("showFrame");
            mBitMapImage = createBitmapFromYuvFrame(yuvFrame, yuv_width, yuv_height);
           // canvas.drawBitmap(mBitMapImage, null, new RectF(0, 0, holder.getSurfaceFrame().width(),  holder.getSurfaceFrame().height()), mPaint);

            if (canvas == null) {
                //     Log.e(TAG, "Cannot draw onto the canvas as it's null");
            } else {
                int canvas_width = holder.getSurfaceFrame().width();
                int canvas_height = holder.getSurfaceFrame().height();
                updateCanvas(canvas, mBitMapImage, canvas_width, canvas_height);
            //    showToast(" UpdateCanvas"  + mBitMapImage);
            }

        }

        mBytes = null;
        mBitMapImage.recycle();

    }


    class CanvasThread extends Thread {
        private SurfaceHolder surfaceHolder;
        private boolean run = false;

        public CanvasThread(SurfaceHolder surfaceHolder) {
            this.surfaceHolder = surfaceHolder;
        }

        public void setRunning(boolean run) {
            this.run = run;
        }

        public SurfaceHolder getSurfaceHolder() { return surfaceHolder; }

        @Override
        public void run() {
            Canvas c;
            int x=0;
            while (run) {
               c = null;
                timeNow = System.currentTimeMillis();
                timeDelta = timeNow - timePrevFrame;
                if (timeDelta < 16) {
                   try {
                        Thread.sleep(16 - timeDelta);
                    } catch (InterruptedException e) {
                    }
                }
                //showToast(" draw");
               timePrevFrame = System.currentTimeMillis();
                try {
                    c = surfaceHolder.lockCanvas(null);
                    synchronized (surfaceHolder) {
                        if (c != null) {
                            draw(c);
                           // surfaceHolder.unlockCanvasAndPost(c);
                        }
                    }
                } finally {
                    if (c != null) {
                        try {
                            surfaceHolder.unlockCanvasAndPost(c);
                        } catch (Exception e) {Log.e(TAG, "surface Unlock" + e.getMessage());}
                    }
                }
            }
        }
    }




    }




