package me.saturnism.micrograd

import kotlin.math.pow
import kotlin.math.tanh

enum class Operation {
    NONE, ADD, MUL, POW, TANH,
}

typealias BackwardLambda = () -> Unit

class Value(var data: Double, val op: Operation = Operation.NONE, val prev: Set<Value> = setOf()) {
    var grad = 0.0
        private set

    private var backwardLambda: BackwardLambda? = null

    operator fun plus(other: Value): Value {
        val out = Value(this.data + other.data, Operation.ADD, setOf(this, other))
        out.backwardLambda = {
            this.grad += out.grad
            other.grad += out.grad
        }
        return out
    }

    operator fun plus(other: Number) = this + Value(other.toDouble())

    operator fun times(other: Value): Value {
        val out = Value(this.data * other.data, Operation.MUL, setOf(this, other))
        out.backwardLambda = {
            this.grad += other.data * out.grad
            other.grad += this.data * out.grad
        }
        return out
    }

    operator fun times(other: Number) = this * Value(other.toDouble())

    operator fun minus(other: Value) = this + other * -1.0
    operator fun minus(other: Number) = this - Value(other.toDouble())

    fun pow(n: Int): Value {
        val out = Value(this.data.pow(n), Operation.POW, setOf(this))
        out.backwardLambda = {
            this.grad += (n * this.data.pow(n - 1)) * out.grad
        }
        return out
    }

    fun tanh(): Value {
        val out = Value(tanh(this.data), Operation.TANH, setOf(this))
        out.backwardLambda = {
            this.grad += (1 - out.data.pow(2)) * out.grad
        }
        return out
    }

    internal fun topoOrdered(): List<Value> {
        val topo = mutableListOf<Value>()
        val visited = mutableSetOf<Value>()

        topoOrdered(this, visited, topo)

        return topo
    }

    private fun topoOrdered(v: Value, visited: MutableSet<Value>, topo: MutableList<Value>) {
        if (v !in visited) {
            visited += v
            for (p in v.prev) {
                topoOrdered(p, visited, topo)
            }
            topo += v
        }
    }

    fun zeroGrad() {
        this.grad = 0.0
        val topo = topoOrdered()
        for (v in topo) {
            v.grad = 0.0
        }
    }

    fun backward() {
        this.grad = 1.0
        val topo = topoOrdered()
        for (v in topo.reversed()) {
            v.backwardLambda?.invoke()
        }
    }

    fun applyGradient(epsilon: Double) {
        this.data += epsilon * this.grad
    }
}
