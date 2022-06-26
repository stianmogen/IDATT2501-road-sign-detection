package com.example.myapplication.model;

import android.graphics.Bitmap;

public class Validate_model {

    // the resource ID for the imageView
    private Bitmap imageId;

    //The yes button
    private int yesButtonId;

    //The no button
    private int noButtonId;

    // TextView 1
    private String textview;



    // create constructor to set the values for all the parameters of the each single view
    public Validate_model(Bitmap NumbersImageId, String textview) {
        imageId = NumbersImageId;
        this.textview = textview;
    }

    // getter method for returning the ID of the imageview
    public Bitmap getimageId() {
        return imageId;
    }

    // getter method for returning the ID of the TextView 1
    public String getTextview() {
        return textview;
    }

    public void setTextview(String input){
        this.textview = input;
    }

    public void setimageId(Bitmap imageId) {
        this.imageId = imageId;
    }

    public int getYesButtonId() {
        return yesButtonId;
    }

    public void setYesButtonId(int yesButtonId) {
        this.yesButtonId = yesButtonId;
    }

    public int getNoButtonId() {
        return noButtonId;
    }

    public void setNoButtonId(int noButtonId) {
        this.noButtonId = noButtonId;
    }
}
