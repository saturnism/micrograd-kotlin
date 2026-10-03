package me.saturnism.micrograd

import kotlin.random.Random

class Neuron(val nin: Int, val rand: Random = Random.Default) {
    private val w = List(nin) { Value(rand.nextDouble(-1.0, 1.0)) }
    private val b = Value(0.0)

    operator fun invoke(x: List<Value>) = w.zip(x)
        .map { (wi, xi) -> wi * xi }
        .reduce { acc, value -> acc + value }
        .plus(b)
        .tanh()
    operator fun invoke(x: DoubleArray) = this(x.map { Value(it) })

    fun parameters() = w + b

    fun applyGradient(epsilon: Double) {
        parameters().forEach { it.applyGradient(epsilon) }
    }
}

class Layer(val nin: Int, val nout: Int, val rand: Random = Random.Default,
) {
    private val neurons = List(nout) { Neuron(nin, rand) }

    operator fun invoke(x: List<Value>) = this.neurons.map { it(x) }
    operator fun invoke(x: DoubleArray) = this(x.map { Value(it) })

    fun parameters() = neurons.flatMap { it.parameters() }

    fun applyGradient(epsilon: Double) {
        neurons.forEach { it.applyGradient(epsilon) }
    }
}

class MLP(val nin: Int, val nouts: IntArray, val rand: Random = Random.Default,
) {
    private val sizes = intArrayOf(nin) + nouts
    private val layers = List(nouts.size) { i -> Layer(sizes[i], sizes[i + 1], rand) }

    operator fun invoke(x: List<Value>): List<Value> {
        var lx = x
        for (layer in layers) { lx = layer(lx) }
        return lx
    }

    operator fun invoke(x: DoubleArray) = this(x.map { Value(it) })

    fun parameters() = layers.flatMap { it.parameters() }

    fun applyGradient(epsilon: Double) {
        layers.forEach { it.applyGradient(epsilon) }
    }
}
