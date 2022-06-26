package com.example.myapplication.util;

import android.content.Context;
import android.graphics.Bitmap;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Spinner;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

import com.example.myapplication.PredictionValidation;
import com.example.myapplication.R;
import com.example.myapplication.TrafficClasses;
import com.example.myapplication.model.Validate_model;

import java.util.ArrayList;

public class ValidateViewAdapter extends ArrayAdapter<Validate_model> {

    public String item;
    String predictionString;

    // invoke the suitable constructor of the ArrayAdapter class
    public ValidateViewAdapter(@NonNull Context context, ArrayList<Validate_model> arrayList) {

        // pass the context and arrayList for the super
        // constructor of the ArrayAdapter class
        super(context, 0, arrayList);
    }


    @NonNull
    @Override
    public View getView(int position, @Nullable View convertView, @NonNull ViewGroup parent) {

        // convertView which is recyclable view
        View currentItemView = convertView;

        // of the recyclable view is null then inflate the custom layout for the same
        if (currentItemView == null) {
            currentItemView = LayoutInflater.from(getContext()).inflate(R.layout.custom_list_view, parent, false);
        }

        // get the position of the view from the ArrayAdapter
        Validate_model currentNumberPosition = getItem(position);

        // then according to the position of the view assign the desired image for the same
        ImageView numbersImage = currentItemView.findViewById(R.id.imageView);
        assert currentNumberPosition != null;
        numbersImage.setImageBitmap(currentNumberPosition.getimageId());

        //Get yes button
        Button yesButton = currentItemView.findViewById(R.id.yesButton);


        //Get no button
        Button noButton = currentItemView.findViewById(R.id.noButton);

        //Get cancel button
        Button cancelButton = currentItemView.findViewById(R.id.cancel_button);

        //Get save button
        Button saveButton = currentItemView.findViewById(R.id.saveButton);

        //Get delete button
        Button deleteButton = currentItemView.findViewById(R.id.deleteButton);

        // then according to the position of the view assign the desired TextView 1 for the same
        TextView textView1 = currentItemView.findViewById(R.id.textView);
        textView1.setText(currentNumberPosition.getTextview());

        //Get and init spinner with list off all traffic signs
        Spinner spinner = (Spinner) currentItemView.findViewById(R.id.spinner);
        initSpinner(spinner);


        // All the on click actions for the different buttons.

        yesButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                noButton.setVisibility(View.INVISIBLE);
                yesButton.setVisibility(View.INVISIBLE);
                deleteButton.setVisibility(View.INVISIBLE);
                textView1.setText("Thank you!");
                Bitmap bitmap = currentNumberPosition.getimageId();
                delete(bitmap);
                //TODO Add logic for properly storing the collected data in the future
            }
        });

        noButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                noButton.setVisibility(View.INVISIBLE);
                yesButton.setVisibility(View.INVISIBLE);
                deleteButton.setVisibility(View.INVISIBLE);
                cancelButton.setVisibility(View.VISIBLE);
                saveButton.setVisibility(View.VISIBLE);
                predictionString = textView1.getText().toString();
                textView1.setText("Please choose the correct sign you see from the list");
                spinner.setVisibility(View.VISIBLE); }
        });


        saveButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                saveButton.setVisibility(View.INVISIBLE);
                cancelButton.setVisibility(View.INVISIBLE);
                spinner.setVisibility(View.INVISIBLE);
                deleteButton.setVisibility(View.INVISIBLE);
                textView1.setText("Thank you!");
                Bitmap bitmap = currentNumberPosition.getimageId();
                delete(bitmap);

                //TODO Add logic for properly storing the collected data in the future
            }
        });

        cancelButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                saveButton.setVisibility(View.INVISIBLE);
                cancelButton.setVisibility(View.INVISIBLE);
                noButton.setVisibility(View.VISIBLE);
                yesButton.setVisibility(View.VISIBLE);
                deleteButton.setVisibility(View.VISIBLE);
                spinner.setVisibility(View.INVISIBLE);
                textView1.setText(predictionString);
            }
        });

        deleteButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                noButton.setVisibility(View.INVISIBLE);
                yesButton.setVisibility(View.INVISIBLE);
                deleteButton.setVisibility(View.INVISIBLE);
                textView1.setText("The image has been deleted");
                Bitmap bitmap = currentNumberPosition.getimageId();
                delete(bitmap);
            }
        });

        // then return the recyclable view
        return currentItemView;
    }

    // Delete a given prediction from list. Either after manually deleting, or validating it.
    // It's too keep us from validating same picture twice
    public void delete(Bitmap bitmap){
        int index = PredictionValidation.signImageList.indexOf(bitmap);
        PredictionValidation.signPredictionList.remove(index);
        PredictionValidation.signImageList.remove(index);
        PredictionValidation.geoLocationList.remove(index);
    }

    // Drop down list with all traffic
    public void initSpinner(Spinner spinner){

        // Set spinner initial element to nothing
        int initialPosition=spinner.getSelectedItemPosition();
        spinner.setSelection(initialPosition, false); //clear selection

        // Spinner click listener
        spinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                // On selecting a spinner item
                 item = parent.getItemAtPosition(position).toString();
            }

            @Override
            public void onNothingSelected(AdapterView<?> parent) {
                //Do nothing
            }
        });

        // Creating adapter for spinner
        ArrayAdapter<String> dataAdapter = new ArrayAdapter<String>(getContext(), android.R.layout.simple_spinner_item, TrafficClasses.TRAFFIC_CLASSES);

        // Drop down layout style - list view with radio button
        dataAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);

        // attaching data adapter to spinner
        spinner.setAdapter(dataAdapter);
    }
}
