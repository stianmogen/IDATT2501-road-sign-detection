package com.example.myapplication;

import java.util.*;

import android.graphics.Bitmap;
import android.os.Bundle;
import android.view.Menu;
import android.view.MenuItem;
import android.widget.ListView;


import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;

import com.example.myapplication.model.Validate_model;
import com.example.myapplication.util.ValidateViewAdapter;

/**
 * Prediction Validation Class for handling model prediction from the user interface
 */
public class PredictionValidation extends AppCompatActivity {

    ArrayList<Validate_model> validateList = new ArrayList<>();
    static public ArrayList<String> signPredictionList = new ArrayList<>();
    static public ArrayList<Bitmap> signImageList = new ArrayList<>();
    static public ArrayList<String> geoLocationList = new ArrayList<>();

    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.prediction_validation_activity);

        Toolbar toolbar = (Toolbar) findViewById(R.id.tool_bar);
        setSupportActionBar(toolbar);

        //Adds all predictions into the validation list for display in interface
        if(!(signPredictionList.size() < 1)) {
            for (int i = 0; i < signPredictionList.size(); i++) {
                //Creates a single string for use in textView
                String signPredict = signPredictionList.get(i) + "\n" + geoLocationList.get(i);
                validateList.add(new Validate_model(signImageList.get(i), signPredict));
            }

            ValidateViewAdapter validateArrayAdapter = new ValidateViewAdapter(this, validateList);

            // create the instance of the ListView to set the numbersViewAdapter
            ListView validateListView = findViewById(R.id.listView);

            // set the numbersViewAdapter for ListView
            validateListView.setAdapter(validateArrayAdapter);

        }
    }

    @Override
    public boolean onCreateOptionsMenu( Menu menu ) {
        getMenuInflater().inflate(R.menu.validation_menu, menu);
        return super.onCreateOptionsMenu(menu);
    }

    @Override
    public boolean onOptionsItemSelected( @NonNull MenuItem item ) {
        finish();
        return super.onOptionsItemSelected(item);
    }

}
