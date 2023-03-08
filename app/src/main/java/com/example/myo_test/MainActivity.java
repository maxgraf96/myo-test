package com.example.myo_test;

import androidx.appcompat.app.AppCompatActivity;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;
import org.tensorflow.lite.nnapi.NnApiDelegate;
import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.gpu.GpuDelegate;

import android.app.Activity;
import android.content.res.AssetFileDescriptor;
import android.os.Build;
import android.os.Bundle;
import android.util.Log;
import android.widget.TextView;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import eu.darken.myolib.Myo;
import eu.darken.myolib.MyoCmds;
import eu.darken.myolib.MyoConnector;
import eu.darken.myolib.msgs.MyoMsg;

public class MainActivity extends AppCompatActivity {

    TextView textView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        textView = (TextView) findViewById(R.id.textView);

        // --------------------------- Myo ---------------------------
        MyoConnector connector = new MyoConnector(MainActivity.this);
        connector.scan(5000, myos -> {
            if(myos.isEmpty())
                return;

            Myo myo = myos.get(0);
            myo.connect();
            myo.writeUnlock(MyoCmds.UnlockType.HOLD, new Myo.MyoCommandCallback() {
                @Override
                public void onCommandDone(Myo myo, MyoMsg msg) {
                    myo.writeVibrate(MyoCmds.VibrateType.LONG, null);
                }
            });
        });


        // --------------------------- TFlite ---------------------------
        // NNAPI delegate
        Interpreter.Options options = (new Interpreter.Options());
        NnApiDelegate nnApiDelegate = null;
        // Initialize interpreter with NNAPI delegate for Android Pie or above
        if(Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            nnApiDelegate = new NnApiDelegate();
            options.addDelegate(nnApiDelegate);
        }

        // Initialize TFLite interpreter
        try {
            Interpreter tfLite = new Interpreter(loadModelFile(MainActivity.this), options);
            textView.setText("TFlite initialized");

            // Print signature list
            List<String> signatureList = Arrays.asList(tfLite.getSignatureKeys());
            Log.d("me", "Signature list: ");
            // Print hello to logcat

            for(String signature : signatureList) {
                Log.d("me", signature);
            }

            // Run inference
            // ...

            // Create random tensor with shape [1, 100, 16] and type FLOAT32
            float[][][] input1 = new float[1][100][16];
            for(int i = 0; i < 1; i++) {
                for(int j = 0; j < 100; j++) {
                    for(int k = 0; k < 16; k++) {
                        input1[i][j][k] = (float) Math.random();
                    }
                }
            }
            float[][][] input2 = new float[1][100][16];
            for(int i = 0; i < 1; i++) {
                for(int j = 0; j < 100; j++) {
                    for(int k = 0; k < 16; k++) {
                        input1[i][j][k] = (float) Math.random();
                    }
                }
            }
            Map<String, Object> inputs = new HashMap<>();
            inputs.put("input", input1);
            inputs.put("tgt", input2);
            Map<String, Object> outputs = new HashMap<>();
            float[][][] output = new float[1][100][16];
            outputs.put("output", output);

            // Time runs
            int runs = 1000;
            long startTime = System.nanoTime();

            for(int i = 0; i < runs; i++) {
                tfLite.runSignature(inputs, outputs);
            }
            long endTime = System.nanoTime();
            long duration = (endTime - startTime) / 1000000;
            float durF = (float) duration / (float) runs;
            Log.d("me", "Average time over " + runs + " runs: " + durF + "ms");

            // Print output
            for(int i = 0; i < 1; i++) {
                for(int j = 0; j < 1; j++) {
                    for(int k = 0; k < 16; k++) {
                        Log.d("me", "output[" + i + "][" + j + "][" + k + "] = " + output[i][j][k]);
                    }
                }
            }

            // Unload delegate
            tfLite.close();

            if(null != nnApiDelegate) {
                nnApiDelegate.close();
            }
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    /** Memory-map the model file in Assets. */
    private MappedByteBuffer loadModelFile(Activity activity) throws IOException {
        AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(getModelPath());
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    private String getModelPath() {
        // Load model from assets
        return "model_hu_2022.tflite";
    }
}