package com.example.noblind

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.view.Surface
import android.view.WindowManager

class OrientationHelper(
    private val context: Context,
    private val speak: (String) -> Unit // função de TTS vinda de fora
) : SensorEventListener {

    private val sensorManager = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    private val rotationSensor = sensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)
    private var lastSpokenTime = 0L
    private var lastMessage = ""

    var allowDetection = false
        private set

    fun startListening() {
        sensorManager.registerListener(this, rotationSensor, SensorManager.SENSOR_DELAY_UI)
    }

    fun stopListening() {
        sensorManager.unregisterListener(this)
    }

    override fun onSensorChanged(event: SensorEvent) {
        val rotationMatrixAdjusted = FloatArray(9)
        val rotationMatrix = FloatArray(9)
        SensorManager.getRotationMatrixFromVector(rotationMatrix, event.values)

        // Alterações
        val windowManager = context.getSystemService(Context.WINDOW_SERVICE) as WindowManager
        val rotation = windowManager.defaultDisplay.rotation

        when(rotation)
        {
            Surface.ROTATION_0 -> SensorManager.remapCoordinateSystem(
                rotationMatrix,
                SensorManager.AXIS_X, SensorManager.AXIS_Z,
                rotationMatrixAdjusted
            )
            Surface.ROTATION_90 -> SensorManager.remapCoordinateSystem(
                rotationMatrix,
                SensorManager.AXIS_Z, SensorManager.AXIS_MINUS_X,
                rotationMatrixAdjusted
            )
            Surface.ROTATION_180 -> SensorManager.remapCoordinateSystem(
                rotationMatrix,
                SensorManager.AXIS_MINUS_X, SensorManager.AXIS_Z,
                rotationMatrixAdjusted
            )
            Surface.ROTATION_270 -> SensorManager.remapCoordinateSystem(
                rotationMatrix,
                SensorManager.AXIS_MINUS_Z, SensorManager.AXIS_X,
                rotationMatrixAdjusted
            )
            else -> System.arraycopy(rotationMatrix, 0, rotationMatrixAdjusted,0,9)
        }


        val orientationAngles = FloatArray(3)
        SensorManager.getOrientation(rotationMatrixAdjusted, orientationAngles)
        val pitch = Math.toDegrees(orientationAngles[1].toDouble()).toFloat()

        // Verifica se está dentro da posição ideal
        allowDetection = pitch in -100f..-80f

        val now = System.currentTimeMillis()
        var message:String? = null

        if(pitch in -10f..10f)
        {
            allowDetection = true
            if(lastMessage!="Posição correta")
            {
                message = "Posição correta"
            }
        }
        else
        {
            allowDetection = false
            message = when{
                pitch < -10f -> "Incline o celular um pouco para baixo."
                pitch > 10f -> "Incline o celular um pouco para cima."
                else -> null
            }
        }

        // Só falar se tiver passado o tempo ou se a mensagem mudou
        if(message!=null&&(message!=lastMessage||now-lastSpokenTime > 3000))
        {
            speak(message)
            lastSpokenTime = now
            lastMessage = message
        }

    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}
}