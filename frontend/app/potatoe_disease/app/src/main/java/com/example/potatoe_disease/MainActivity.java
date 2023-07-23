package com.example.potatoe_disease;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import org.tensorflow.lite.Interpreter;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

public class MainActivity extends AppCompatActivity {

    private static final int CAMERA_PERMISSION_REQUEST_CODE = 200;
    private static final int CAMERA_REQUEST_CODE = 201;
    private static final int GALLERY_PERMISSION_REQUEST_CODE = 202;
    private static final int GALLERY_REQUEST_CODE = 203;

    private ImageView imageView;
    private TextView predictionTextView;

    private Interpreter tflite;
    private ByteBuffer imgData;
    private String[] classNames = {"Early Blight", "Late Blight", "Healthy"};

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        imageView = findViewById(R.id.imageView);
        predictionTextView = findViewById(R.id.predictionTextView);

        Button captureButton = findViewById(R.id.captureButton);
        captureButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (ContextCompat.checkSelfPermission(MainActivity.this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                    captureImage();
                } else {
                    ActivityCompat.requestPermissions(MainActivity.this, new String[]{Manifest.permission.CAMERA}, CAMERA_PERMISSION_REQUEST_CODE);
                }
            }
        });

        Button galleryButton = findViewById(R.id.galleryButton);
        galleryButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (ContextCompat.checkSelfPermission(MainActivity.this, Manifest.permission.READ_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED) {
                    selectFromGallery();
                } else {
                    ActivityCompat.requestPermissions(MainActivity.this, new String[]{Manifest.permission.READ_EXTERNAL_STORAGE}, GALLERY_PERMISSION_REQUEST_CODE);
                }
            }
        });

        try {
            tflite = new Interpreter(loadModelFile());
        } catch (IOException e) {
            e.printStackTrace();
        }

        // Set the correct size for imgData based on the model's input shape
        int inputImageWidth = 256;
        int inputImageHeight = 256;
        int channels = 3;
        imgData = ByteBuffer.allocateDirect(inputImageWidth * inputImageHeight * channels * 4); // 4 bytes for each float
        imgData.order(ByteOrder.nativeOrder());
    }



    private ByteBuffer loadModelFile() throws IOException {
        InputStream inputStream = getAssets().open("output_model2.tflite");
        int fileSize = inputStream.available();
        byte[] buffer = new byte[fileSize];
        inputStream.read(buffer);
        inputStream.close();
        ByteBuffer modelBuffer = ByteBuffer.allocateDirect(fileSize);
        modelBuffer.order(ByteOrder.nativeOrder());
        modelBuffer.put(buffer);
        modelBuffer.rewind();
        return modelBuffer;
    }

    private void captureImage() {
        Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        if (intent.resolveActivity(getPackageManager()) != null) {
            startActivityForResult(intent, CAMERA_REQUEST_CODE);
        }
    }

    private void selectFromGallery() {
        if (ContextCompat.checkSelfPermission(MainActivity.this, Manifest.permission.READ_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED) {
            openGallery();
        } else {
            ActivityCompat.requestPermissions(MainActivity.this, new String[]{Manifest.permission.READ_EXTERNAL_STORAGE}, GALLERY_PERMISSION_REQUEST_CODE);
        }
    }

    private void openGallery() {
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType("image/*");
        if (intent.resolveActivity(getPackageManager()) != null) {
            startActivityForResult(intent, GALLERY_REQUEST_CODE);
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == CAMERA_REQUEST_CODE && resultCode == RESULT_OK) {
            Bitmap imageBitmap = (Bitmap) data.getExtras().get("data");
            processImage(imageBitmap);
        } else if (requestCode == GALLERY_REQUEST_CODE && resultCode == RESULT_OK) {
            try {
                if (data != null && data.getData() != null) {
                    InputStream inputStream = getContentResolver().openInputStream(data.getData());
                    Bitmap imageBitmap = BitmapFactory.decodeStream(inputStream);
                    processImage(imageBitmap);
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    private void processImage(Bitmap imageBitmap) {
        imageView.setImageBitmap(imageBitmap);

        Bitmap scaledBitmap = Bitmap.createScaledBitmap(imageBitmap, 256, 256, true);
        convertBitmapToByteBuffer(scaledBitmap);

        float[][] result = new float[1][3];
        tflite.run(imgData, result);

        String predictedClass = classNames[argmax(result[0])];
        float confidence = result[0][argmax(result[0])];

        predictionTextView.setText("Prediction: " + predictedClass + "\nConfidence: " + confidence);
    }

    private void convertBitmapToByteBuffer(Bitmap bitmap) {
        if (imgData == null) {
            return;
        }

        imgData.rewind();
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        int[] intValues = new int[width * height];
        bitmap.getPixels(intValues, 0, width, 0, 0, width, height);

        int pixel = 0;
        for (int i = 0; i < width; ++i) {
            for (int j = 0; j < height; ++j) {
                int value = intValues[pixel++];
                imgData.putFloat((float) (value & 0xFF) / 255.0f); // Blue channel
                imgData.putFloat((float) ((value >> 8) & 0xFF) / 255.0f); // Green channel
                imgData.putFloat((float) ((value >> 16) & 0xFF) / 255.0f); // Red channel
            }
        }
    }







    private int argmax(float[] array) {
        int maxIndex = 0;
        float maxValue = array[0];
        for (int i = 1; i < array.length; i++) {
            if (array[i] > maxValue) {
                maxIndex = i;
                maxValue = array[i];
            }
        }
        return maxIndex;
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == CAMERA_PERMISSION_REQUEST_CODE && grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            captureImage();
        } else if (requestCode == GALLERY_PERMISSION_REQUEST_CODE && grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            openGallery();
        } else {
            Toast.makeText(MainActivity.this, "Permission denied", Toast.LENGTH_SHORT).show();
        }
    }
}
