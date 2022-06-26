package com.example.myapplication;

import android.content.Context;
import android.graphics.Rect;
import android.hardware.Camera;
import android.util.AttributeSet;
import android.view.MotionEvent;

import org.opencv.android.JavaCameraView;

import java.util.ArrayList;

/**
 * Custom class that implements autofocus on touch events.
 */
public class MyCameraView extends JavaCameraView {

    private final Camera.AutoFocusCallback autoFocusCallback = (success, camera) -> {
    };

    private final int focusAreaSize = 100;

    public MyCameraView(Context context, AttributeSet attrs) {
        super(context, attrs);
        mCamera = Camera.open();
    }


    /**
     * Method for setting the focus area and metering area of the camera on a touch event.
     * Predicts bounds of camera and finds point corresponding to touch event.
     * @param event event of the touch, contains x and y coordinates
     * @param toolbar_offset offset of the toolbar on the device
     */
    public void focusOnTouch(MotionEvent event, int toolbar_offset) {
        float x = event.getX();
        float y = event.getY();

        int offset_top = getRootWindowInsets().getStableInsetTop() + toolbar_offset;

        int view_width = this.getWidth();

        int cam_width = mFrameWidth;
        int cam_height = mFrameHeight;

        if (mScale > 0) {
            cam_width = (int) (cam_width*mScale);
            cam_height = (int) (cam_height*mScale);
        }


        int offset_x = (view_width-cam_width)/2;

        x -= offset_x;
        y -= offset_top;

        if (x < 0) x = 0;
        if (x > cam_width) x = cam_width;

        if (y < 0) y = 0;
        if (y > cam_height) y = cam_height;

        if (mCamera != null) {
            Camera.Parameters params = mCamera.getParameters();


            if (params.getMaxNumFocusAreas() > 0){
                mCamera.cancelAutoFocus();
                if (params.getMaxNumMeteringAreas() > 0){
                    Rect meteringRect = calculateTapArea(x, y, 1.5f, cam_width, cam_height);
                    ArrayList<Camera.Area> meteringAreas = new ArrayList<>();
                    meteringAreas.add(new Camera.Area(meteringRect, 1000));
                    params.setMeteringAreas(meteringAreas);
                }
                Rect focusRect = calculateTapArea(x, y, 1f, cam_width, cam_height);

                ArrayList<Camera.Area> focusAreas = new ArrayList<>();
                focusAreas.add(new Camera.Area(focusRect, 1000));

                if (params.getSupportedFocusModes().contains(
                        Camera.Parameters.FOCUS_MODE_AUTO)) {
                    params.setFocusMode(Camera.Parameters.FOCUS_MODE_AUTO);
                }
                params.setFocusAreas(focusAreas);
                mCamera.setParameters(params);
                mCamera.autoFocus(autoFocusCallback);
            }
        }
    }


    /**
     * Method for calculating what area on to focus based on coordinates on camera.
     * @param x coordinate for center point of touch
     * @param y coordinate for center point of touch
     * @param coefficient for scaling tap area
     * @param width width of screen
     * @param height height of screen
     * @return Rect of where focus area is located.
     */
    private Rect calculateTapArea(float x, float y, float coefficient, int width, int height) {
        int areaSize = Float.valueOf(focusAreaSize * coefficient).intValue();
        System.out.println(x + " " + y);
        float left = clamp(x - focusAreaSize / 2f, width - focusAreaSize);
        float top = clamp(y - focusAreaSize / 2f, height - focusAreaSize);


        return new Rect(
                (int) (left/width*2000-1000),
                (int) (top/height*2000-1000),
                (int) ((left + areaSize)/width*2000-1000),
                (int) ((top + areaSize)/height*2000-1000));
    }


    /**
     * Helper method for not going outside screen bounds
     * @param x value
     * @param max maximum limit of value
     * @return value between 0 and max value
     */
    private float clamp(float x, float max) {
        if (x > max) {
            return max;
        }
        return Math.max(x, 0);
    }
}
