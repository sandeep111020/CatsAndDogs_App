package com.example.catsdogs;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.drawable.BitmapDrawable;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import com.example.catsdogs.ml.Model;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.List;

public class MainActivity extends AppCompatActivity {

    private int mInputSize = 224;
    private String mModelPath= "converted_model.tflite";
    private  String mLabelPath = "label.txt";


    private  ImageView imgview;
    private Button select, predict;
    protected TextView textView;
    private  Bitmap bitmap;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        imgview = (ImageView) findViewById(R.id.imageview);
        select = (Button) findViewById(R.id.select);
        predict = (Button) findViewById(R.id.predict);
        textView = (TextView) findViewById(R.id.textview);

        select.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
                intent.setType("image/*");
                startActivityForResult(intent, 100);

            }
        });
        predict.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                bitmap = Bitmap.createScaledBitmap(bitmap, 128, 128, true);

                try {
                    Model model = Model.newInstance(getApplicationContext());

                    // Creates inputs for reference.
                    TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 128, 128, 3}, DataType.FLOAT32);

                    TensorImage tensorImage = new TensorImage(DataType.FLOAT32);
                    tensorImage.load(bitmap);
                    ByteBuffer byteBuffer = tensorImage.getBuffer();

                    inputFeature0.loadBuffer(byteBuffer);

                    // Runs model inference and gets result.
                    Model.Outputs outputs = model.process(inputFeature0);
                    TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

                    // Releases model resources if no longer used.
                    model.close();

                    if(outputFeature0.getFloatArray()[0]==0)
                        textView.setText("This is a dog");
                    else
                        textView.setText(" This is a cat");



                    //textView.setText(outputFeature0.getFloatArray()[0] + "\n"+outputFeature0.getFloatArray()[1]);


                } catch (IOException e) {
                    // TODO Handle the exception
                }

            }
        });

       /* try {
            initClassifier();
            initViews();
        }catch (IOException e){
            e.printStackTrace();
        }*/

    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if(requestCode == 100)
        {
            imgview.setImageURI(data.getData());

            Uri uri = data.getData();
            try {
                bitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), uri);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }
   /* private void initClassifier() throws IOException{
        classifier = new Classifier(getAssets(),mModelPath,mLabelPath,mInputSize);
    }
    private  void initViews()
    {
        findViewById(R.id.iv1).setOnClickListener(this);
        findViewById(R.id.iv2).setOnClickListener(this);
        findViewById(R.id.iv3).setOnClickListener(this);
        findViewById(R.id.iv4).setOnClickListener(this);
    }

    @Override
    public void onClick(View v) {
        Bitmap bitmap =((BitmapDrawable)((ImageView)v).getDrawable()).getBitmap();
        List<Classifier.Recognition> result = classifier.reconizeImage(bitmap);
        Toast.makeText(this, result.get(0).toString(), Toast.LENGTH_SHORT).show();;

    }*/
}