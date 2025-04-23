package com.example.noblind

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.graphics.RectF
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.speech.tts.TextToSpeech
import android.util.Log
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.AspectRatio
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.example.noblind.Constants.CLASSIFICATION_DELAY
import com.example.noblind.Constants.CLOTHING_LABELS_PATH
import com.example.noblind.Constants.CLOTHING_MODEL_PATH
import com.example.noblind.Constants.LABELS_PATH
import com.example.noblind.Constants.MODEL_PATH
import com.example.noblind.databinding.ActivityMainBinding
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity(), Detector.DetectorListener, ClothingClassifier.ClassifierListener {
    private lateinit var binding: ActivityMainBinding
    private val isFrontCamera = false

    private var preview: Preview? = null
    private var imageAnalyzer: ImageAnalysis? = null
    private var camera: Camera? = null
    private var cameraProvider: ProcessCameraProvider? = null
    private var detector: Detector? = null
    private var lastSpokenObjects = setOf<String>()
    private var lastDetectionTime = 0L
    private var isPersonConsistentlyDetected = false
    private val stabilizationDelay = 3000L
    // Camera
    private var allowDetection = true

    // Classificador de roupas
    private val colorDetector = ColorDetector()
    private var clothingClassifier: ClothingClassifier? = null
    private val handler = Handler(Looper.getMainLooper())
    private var isClassificationScheduled = false
    private var currentFrameBitmap: Bitmap? = null
    private var currentDetectedBoxes = listOf<BoundingBox>()

    private lateinit var cameraExecutor: ExecutorService
    private lateinit var tts: TextToSpeech
    //Camera
    private lateinit var orientationHelper: OrientationHelper

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        cameraExecutor = Executors.newSingleThreadExecutor()

        cameraExecutor.execute {
            detector = Detector(baseContext, MODEL_PATH, LABELS_PATH, this)
            clothingClassifier = ClothingClassifier(baseContext, CLOTHING_MODEL_PATH, CLOTHING_LABELS_PATH, this)
        }

        if (allPermissionsGranted()) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS)
        }

        bindListeners()

        tts = TextToSpeech(this) { status ->
            if (status == TextToSpeech.SUCCESS) {
                tts.language = Locale("pt", "BR")
            }
        }

        val speakFunction: (String) -> Unit = { text ->
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP) {
                tts.speak(text, TextToSpeech.QUEUE_FLUSH, null, null)
            } else {
                tts.speak(text, TextToSpeech.QUEUE_FLUSH, null)
            }
        }

        orientationHelper = OrientationHelper(this, speakFunction)
        orientationHelper.startListening()
    }

    private fun bindListeners() {
        binding.apply {
            isGpu.setOnCheckedChangeListener { buttonView, isChecked ->
                cameraExecutor.submit {
                    detector?.restart(isGpu = isChecked)
                    clothingClassifier?.restart(isGpu = isChecked)
                }
                if (isChecked) {
                    buttonView.setBackgroundColor(ContextCompat.getColor(baseContext, R.color.orange))
                } else {
                    buttonView.setBackgroundColor(ContextCompat.getColor(baseContext, R.color.gray))
                }
            }
        }
    }

    private fun scheduleClassification() {
        if (!isClassificationScheduled) {
            isClassificationScheduled = true
            binding.classificationStatus.text = "Preparando para classificar roupa..."

            handler.postDelayed({
                if (isPersonConsistentlyDetected) { // Só classifica se a pessoa ainda estiver sendo detectada
                    currentFrameBitmap?.let { bitmap ->
                        val clothingBox = currentDetectedBoxes
                            .firstOrNull { it.clsName == "Pessoa" }
                            ?.let { box ->
                                RectF(box.x1, box.y1, box.x2, box.y2)
                            }

                        tts.speak("Classificando roupa", TextToSpeech.QUEUE_FLUSH, null, null)
                        binding.classificationStatus.text = "Classificando..."

                        clothingClassifier?.classify(bitmap, clothingBox)
                    } ?: run {
                        binding.classificationStatus.text = "Falha na captura de imagem"
                        tts.speak("Falha na captura de imagem", TextToSpeech.QUEUE_FLUSH, null, null)
                        isClassificationScheduled = false
                        isPersonConsistentlyDetected = false
                    }
                } else {
                    isClassificationScheduled = false
                }
            }, CLASSIFICATION_DELAY)
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            cameraProvider = cameraProviderFuture.get()
            bindCameraUseCases()
        }, ContextCompat.getMainExecutor(this))
    }

    private fun bindCameraUseCases() {
        val cameraProvider = cameraProvider ?: throw IllegalStateException("Camera initialization failed.")

        val rotation = binding.viewFinder.display.rotation

        val cameraSelector = CameraSelector
            .Builder()
            .requireLensFacing(CameraSelector.LENS_FACING_BACK)
            .build()

        preview = Preview.Builder()
            .setTargetAspectRatio(AspectRatio.RATIO_4_3)
            .setTargetRotation(rotation)
            .build()

        imageAnalyzer = ImageAnalysis.Builder()
            .setTargetAspectRatio(AspectRatio.RATIO_4_3)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setTargetRotation(binding.viewFinder.display.rotation)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .build()

        imageAnalyzer?.setAnalyzer(cameraExecutor) { imageProxy ->
            val bitmapBuffer =
                Bitmap.createBitmap(
                    imageProxy.width,
                    imageProxy.height,
                    Bitmap.Config.ARGB_8888
                )
            imageProxy.use { bitmapBuffer.copyPixelsFromBuffer(imageProxy.planes[0].buffer) }
            imageProxy.close()

            val matrix = Matrix().apply {
                postRotate(imageProxy.imageInfo.rotationDegrees.toFloat())

                if (isFrontCamera) {
                    postScale(
                        -1f,
                        1f,
                        imageProxy.width.toFloat(),
                        imageProxy.height.toFloat()
                    )
                }
            }

            val rotatedBitmap = Bitmap.createBitmap(
                bitmapBuffer, 0, 0, bitmapBuffer.width, bitmapBuffer.height,
                matrix, true
            )

            // Guardar o frame atual para classificação
            currentFrameBitmap = rotatedBitmap.config?.let { rotatedBitmap.copy(it, true) }

            if (orientationHelper.allowDetection) {
                detector?.detect(rotatedBitmap)
            }
        }

        cameraProvider.unbindAll()

        try {
            camera = cameraProvider.bindToLifecycle(
                this,
                cameraSelector,
                preview,
                imageAnalyzer
            )

            preview?.setSurfaceProvider(binding.viewFinder.surfaceProvider)
        } catch(exc: Exception) {
            Log.e(TAG, "Use case binding failed", exc)
        }
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()) {
        if (it[Manifest.permission.CAMERA] == true) { startCamera() }
    }

    override fun onDestroy() {
        tts.shutdown()
        super.onDestroy()
        detector?.close()
        clothingClassifier?.close()
        cameraExecutor.shutdown()
    }

    override fun onResume() {
        super.onResume()
        if (allPermissionsGranted()){
            startCamera()
        } else {
            requestPermissionLauncher.launch(REQUIRED_PERMISSIONS)
        }
    }

    companion object {
        private const val TAG = "Camera"
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = mutableListOf (
            Manifest.permission.CAMERA
        ).toTypedArray()
    }

    override fun onEmptyDetect() {
        runOnUiThread {
            binding.overlay.clear()
            currentDetectedBoxes = listOf()
            isPersonConsistentlyDetected = false
            lastDetectionTime = 0L
        }
    }

    override fun onDetect(boundingBoxes: List<BoundingBox>, inferenceTime: Long) {
        val currentObjects = boundingBoxes.map { it.clsName }.toSet()
        currentDetectedBoxes = boundingBoxes

        runOnUiThread {
            if (currentObjects != lastSpokenObjects) {
                lastSpokenObjects = currentObjects
                val description = currentObjects.joinToString(", ")
                tts.speak("Detectado: $description", TextToSpeech.QUEUE_FLUSH, null, null)
                // Resetar o timer se a detecção mudar
                if ("Pessoa" in currentObjects) {
                    lastDetectionTime = SystemClock.uptimeMillis()
                    isPersonConsistentlyDetected = false
                } else {
                    isPersonConsistentlyDetected = false
                }
            }
            if ("Pessoa" in currentObjects && !isClassificationScheduled && !isPersonConsistentlyDetected) {
                val currentTime = SystemClock.uptimeMillis()
                if (currentTime - lastDetectionTime >= stabilizationDelay) {
                    isPersonConsistentlyDetected = true
                    scheduleClassification()
                }
            }
            binding.inferenceTime.text = "${inferenceTime}ms"
            binding.overlay.apply {
                setResults(boundingBoxes)
                invalidate()
            }
        }
    }

    // Implementação do callback do classificador
    override fun onClassificationResults(
        results: List<ClothingClassifier.ClassificationResult>,
        inferenceTime: Long
    ) {
        runOnUiThread {
            val topResult = results.firstOrNull()

            // Detectar cor apenas se tivermos uma bounding box
            val color = currentDetectedBoxes.firstOrNull()?.let { box ->
                val clothingBox = RectF(box.x1, box.y1, box.x2, box.y2)
                currentFrameBitmap?.let { bitmap ->
                    colorDetector.detectDominantColor(bitmap, clothingBox)
                }
            } ?: "cor não detectada"

            val resultText = if (topResult != null) {
                val confidencePercent = (topResult.confidence * 100).toInt()
                "Roupa: ${topResult.className} ($confidencePercent%), Cor: $color"
            } else {
                "Nenhuma classificação encontrada"
            }

            binding.classificationStatus.text = resultText

            // Mensagem de voz combinando tipo e cor
            val speechText = if (results.isNotEmpty()) {
                "${topResult?.className} ${color.replace("claro", "de cor clara").replace("escuro", "de cor escura")}"
            } else {
                "Não foi possível classificar a roupa"
            }

            tts.speak(speechText, TextToSpeech.QUEUE_FLUSH, null, null)
            isClassificationScheduled = false
        }
    }
}