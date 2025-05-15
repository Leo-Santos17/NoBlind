package com.example.noblind

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import android.os.SystemClock
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.CastOp
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.BufferedReader
import java.io.IOException
import java.io.InputStream
import java.io.InputStreamReader

/**
 * Um classificador genérico que pode ser usado com diferentes modelos TensorFlow Lite.
 *
 * @param context O contexto da aplicação
 * @param modelPath O caminho para o arquivo do modelo TFLite no diretório assets
 * @param labelPath O caminho para o arquivo de rótulos no diretório assets
 * @param listener O listener que receberá os resultados da classificação
 * @param classifierType O tipo do classificador (usado para diferenciar múltiplos classificadores)
 */
class GenericClassifier(
    private val context: Context,
    private val modelPath: String,
    private val labelPath: String,
    private val listener: ClassifierListener,
    private val classifierType: ClassifierType
) {
    private var interpreter: Interpreter
    private var labels = mutableListOf<String>()
    private var tensorWidth = 0
    private var tensorHeight = 0

    private val imageProcessor = ImageProcessor.Builder()
        .add(NormalizeOp(INPUT_MEAN, INPUT_STANDARD_DEVIATION))
        .add(CastOp(INPUT_IMAGE_TYPE))
        .build()

    init {
        val compatList = CompatibilityList()

        val options = Interpreter.Options().apply {
            if (compatList.isDelegateSupportedOnThisDevice) {
                val delegateOptions = compatList.bestOptionsForThisDevice
                this.addDelegate(GpuDelegate(delegateOptions))
            } else {
                this.setNumThreads(4)
            }
        }

        val model = FileUtil.loadMappedFile(context, modelPath)
        interpreter = Interpreter(model, options)

        val inputShape = interpreter.getInputTensor(0)?.shape()

        if (inputShape != null) {
            tensorWidth = inputShape[1]
            tensorHeight = inputShape[2]

            // Se o formato da entrada for [1, 3, ..., ...]
            if (inputShape[1] == 3) {
                tensorWidth = inputShape[2]
                tensorHeight = inputShape[3]
            }
        }

        try {
            val inputStream: InputStream = context.assets.open(labelPath)
            val reader = BufferedReader(InputStreamReader(inputStream))

            var line: String? = reader.readLine()
            while (line != null && line != "") {
                labels.add(line)
                line = reader.readLine()
            }

            reader.close()
            inputStream.close()
        } catch (e: IOException) {
            e.printStackTrace()
        }
    }

    fun classify(bitmap: Bitmap, boundingBox: RectF? = null) {
        if (tensorWidth == 0 || tensorHeight == 0) return

        var inferenceTime = SystemClock.uptimeMillis()

        // Se tiver um boundingBox, recorte a imagem antes de classificar
        val imageToBeclassified = if (boundingBox != null) {
            val x = (boundingBox.left * bitmap.width).toInt().coerceIn(0, bitmap.width)
            val y = (boundingBox.top * bitmap.height).toInt().coerceIn(0, bitmap.height)
            val width = (boundingBox.width() * bitmap.width).toInt()
                .coerceIn(1, bitmap.width - x)
            val height = (boundingBox.height() * bitmap.height).toInt()
                .coerceIn(1, bitmap.height - y)

            Bitmap.createBitmap(bitmap, x, y, width, height)
        } else {
            bitmap
        }

        val resizedBitmap = Bitmap.createScaledBitmap(
            imageToBeclassified,
            tensorWidth,
            tensorHeight,
            false
        )

        val tensorImage = TensorImage(INPUT_IMAGE_TYPE)
        tensorImage.load(resizedBitmap)
        val processedImage = imageProcessor.process(tensorImage)

        // Output buffer para armazenar os resultados da classificação
        val outputBuffer = TensorBuffer.createFixedSize(intArrayOf(1, labels.size), DataType.FLOAT32)

        interpreter.run(processedImage.buffer, outputBuffer.buffer)

        inferenceTime = SystemClock.uptimeMillis() - inferenceTime

        // Obter os resultados da classificação
        val results = outputBuffer.floatArray

        // Encontrar os N melhores resultados
        val topN = 1 // Número de resultados principais a retornar
        val topResults = mutableListOf<ClassificationResult>()

        for (i in results.indices) {
            topResults.add(ClassificationResult(labels[i], results[i]))
        }

        // Ordenar por confiança e pegar os N melhores
        topResults.sortByDescending { it.confidence }
        val bestResults = topResults.take(topN)

        listener.onClassificationResults(bestResults, inferenceTime, classifierType)
    }

    fun restart(isGpu: Boolean) {
        interpreter.close()

        val options = if (isGpu) {
            val compatList = CompatibilityList()
            Interpreter.Options().apply {
                if (compatList.isDelegateSupportedOnThisDevice) {
                    val delegateOptions = compatList.bestOptionsForThisDevice
                    this.addDelegate(GpuDelegate(delegateOptions))
                } else {
                    this.setNumThreads(4)
                }
            }
        } else {
            Interpreter.Options().apply {
                this.setNumThreads(4)
            }
        }

        val model = FileUtil.loadMappedFile(context, modelPath)
        interpreter = Interpreter(model, options)
    }

    fun close() {
        interpreter.close()
    }

    data class ClassificationResult(
        val className: String,
        val confidence: Float
    )

    /**
     * Interface para receber os resultados da classificação
     */
    interface ClassifierListener {
        fun onClassificationResults(
            results: List<ClassificationResult>,
            inferenceTime: Long,
            classifierType: ClassifierType
        )
    }

    /**
     * Enum para identificar diferentes tipos de classificadores
     */
    enum class ClassifierType {
        CLOTHING,   // Classificador de roupas superior
        BOTTOM,     // Classificador de calças/roupas inferiores
        COLOR,      // Classificador de cores
        OTHER       // Outros tipos de classificadores
    }

    companion object {
        private const val INPUT_MEAN = 0f
        private const val INPUT_STANDARD_DEVIATION = 255f
        private val INPUT_IMAGE_TYPE = DataType.FLOAT32
    }
}