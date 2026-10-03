package me.saturnism.micrograd

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test
import kotlin.random.Random

class NeuralNetworkTest {
    @Test
    fun testNeuron() {
        val n1 = Neuron(1)
        n1(doubleArrayOf(1.0))
        val p1 = n1.parameters()
        assertEquals(2, p1.size)

        val n2 = Neuron(4)
        n2(doubleArrayOf(1.0, 2.0, 3.0, 4.0))
        val p2 = n2.parameters()
        assertEquals(5, p2.size)
    }

    @Test
    fun testLayer() {
        val l1 = Layer(1, 1)
        val o1 = l1(doubleArrayOf(1.0))
        assertEquals(1, o1.size)
        val p1 = l1.parameters()
        assertEquals(2, p1.size)

        val l2 = Layer(2, 2)
        val o2 = l2(doubleArrayOf(1.0, 2.0))
        assertEquals(2, o2.size)
        val p2 = l2.parameters()
        assertEquals(6, p2.size)
    }

    @Test
    fun testMLP() {
        MLP(1, intArrayOf(1))
        MLP(1, intArrayOf(3, 3, 1))
    }

    @Test
    fun testGradientDescent() {
        val xs =
            arrayOf(
                doubleArrayOf(2.0, 3.0, -1.0),
                doubleArrayOf(3.0, -1.0, 0.5),
                doubleArrayOf(0.5, 1.0, -1.0),
                doubleArrayOf(1.0, 1.0, -1.0),
            )
        val ys = doubleArrayOf(1.0, -1.0, -1.0, 1.0)

        var loss = Value(Double.MAX_VALUE)

        // use a static seed to ensure reproducibility
        val mlp = MLP(3, intArrayOf(4, 4, 1), Random(1))

        repeat(500) { k ->
            // forward
            val ypred =
                List(xs.size) { i ->
                    val yp = mlp(xs[i])
                    assertEquals(1, yp.size)
                    yp.first()
                }
            assertEquals(4, ypred.size)
            loss = ys.zip(ypred).map { (ygt, yout) -> (yout - ygt).pow(2) }.reduce { acc, value -> acc + value }

            // backward
            loss.zeroGrad()
            loss.backward()
            mlp.applyGradient(-0.1)

            if (k % 100 == 0) {
                println("$k loss: ${loss.data}")
            }
        }
        assertEquals(0.0, loss.data, 0.001)

        // check the results
        val ypred =
            List(xs.size) { i ->
                mlp(xs[i]).first()
            }

        println(ys.zip(ypred.map { it.data }))

        ys.zip(ypred).forEach { (y, pred) ->
            assertEquals(y, pred.data, 0.02)
        }
    }
}
