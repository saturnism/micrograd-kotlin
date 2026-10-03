package me.saturnism.micrograd

import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test
import kotlin.math.tanh

class EngineTest {
    @Test
    fun testBasics() {
        val x = Value(1.5)
        val y = Value(2.0)
        assertEquals(1.5, x.data)
        assertEquals(2.0, y.data)

        val a = x + y
        assertEquals(3.5, a.data)
        assertEquals(Operation.ADD, a.op)

        val c = x * y
        assertEquals(3.0, c.data)
        assertEquals(Operation.MUL, c.op)

        val d = c.pow(2)
        assertEquals(9.0, d.data)
        assertEquals(Operation.POW, d.op)

        val p = x + 4.0
        assertEquals(5.5, p.data)

        val q = x * 3.0
        assertEquals(4.5, q.data)

        val m = y - 5.0
        assertEquals(-3.0, m.data)

        val r = y.pow(2)
        assertEquals(4.0, r.data)

        val t = x.tanh()
        assertEquals(tanh(x.data), t.data)
        assertEquals(Operation.TANH, t.op)
    }

    @Test
    fun testBackward() {
        val a = Value(1.5)
        val b = Value(2.0)
        val c = a + b
        c.backward()
        assertEquals(1.0, c.grad)
        assertEquals(c.grad, a.grad)
        assertEquals(c.grad, b.grad)

        c.zeroGrad()
        assertEquals(0.0, a.grad)
        assertEquals(0.0, b.grad)
        assertEquals(0.0, c.grad)

        val d = b + b
        d.zeroGrad()
        d.backward()
        assertEquals(1.0, d.grad)
        assertEquals(2.0, b.grad)

        val m = a * b
        m.zeroGrad()
        m.backward()
        assertEquals(2.0, a.grad)
        assertEquals(1.5, b.grad)

        val p = b.pow(3)
        p.zeroGrad()
        p.backward()
        assertEquals(12.0, b.grad)
    }

    @Test
    fun testBackwardChain() {
        val x1 = Value(2.0)
        val x2 = Value(0.0)
        val w1 = Value(-3.0)
        val w2 = Value(1.0)
        val b = Value(6.881373587019543)
        var x1w1 = x1 * w1
        var x2w2 = x2 * w2
        var x1w1x2w2 = x1w1 + x2w2
        var n = x1w1x2w2 + b
        var o = n.tanh()

        assertEquals(0.7071067811865477, o.data)
        o.backward()

        val tolerance = 0.00000001
        assertEquals(1.0, o.grad)
        assertEquals(0.5, n.grad, tolerance)
        assertEquals(0.5, b.grad, tolerance)
        assertEquals(0.5, x1w1x2w2.grad, tolerance)
        assertEquals(0.5, x1w1.grad, tolerance)
        assertEquals(0.5, x2w2.grad, tolerance)
        assertEquals(-1.5, x1.grad, tolerance)
        assertEquals(1.0, w1.grad, tolerance)
        assertEquals(0.5, x2.grad, tolerance)
        assertEquals(0.0, w2.grad, tolerance)

        o.zeroGrad()
        assertEquals(0.0, o.grad)
        assertEquals(0.0, n.grad)
        assertEquals(0.0, b.grad)
        assertEquals(0.0, x1w1x2w2.grad)
        assertEquals(0.0, x1w1.grad)
        assertEquals(0.0, x2w2.grad)
        assertEquals(0.0, x1.grad)
        assertEquals(0.0, w1.grad)
        assertEquals(0.0, x2.grad)
        assertEquals(0.0, w2.grad)

        val eps = 0.1
        repeat(100) {
            o.zeroGrad()
            o.backward()
            w1.applyGradient(-eps)
            w2.applyGradient(-eps)
            b.applyGradient(-eps)

            x1w1 = x1 * w1
            x2w2 = x2 * w2
            x1w1x2w2 = x1w1 + x2w2
            n = x1w1x2w2 + b
            o = n.tanh()
        }
        assertEquals(-1.0, o.data, 0.01)
    }
}
