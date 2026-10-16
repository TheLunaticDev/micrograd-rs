use std::{
    cell::RefCell,
    ops::{Add, Mul},
};

/// Struct that holds the value.
#[derive(Copy, Clone)]
struct Node {
    data: f64,
    grad: f64,
    op: Op,
}

/// Holds all operations supported by micrograd-rs.
#[derive(Debug, Copy, Clone, PartialEq)]
enum Op {
    Leaf,
    Add(usize, usize),
    Mul(usize, usize),
}

/// A sharable object that holds nodes and allows operations.
pub struct Arena {
    nodes: RefCell<Vec<Node>>,
}

impl Default for Arena {
    fn default() -> Self {
        Self::new()
    }
}

impl Arena {
    /// Creates a new empty arena.
    pub fn new() -> Self {
        Arena {
            nodes: RefCell::new(Vec::<Node>::new()),
        }
    }

    /// Creates a new leaf arena node.
    pub fn leaf(&self, data: f64) -> Value<'_> {
        self.push_node(data, Op::Leaf)
    }

    fn push_node(&self, data: f64, op: Op) -> Value<'_> {
        let mut borrow = self.nodes.borrow_mut();
        let idx = borrow.len();
        let node = Node {
            data,
            grad: 0.0,
            op,
        };
        borrow.push(node);
        Value { idx, arena: self }
    }

    pub fn zero_grad(&self) {
        for node in self.nodes.borrow_mut().iter_mut() {
            node.grad = 0.0;
        }
    }
}

#[derive(Clone, Copy)]
pub struct Value<'a> {
    idx: usize,
    arena: &'a Arena,
}

impl Value<'_> {
    fn ensure_same_arena(&self, other: &Value) {
        assert!(
            std::ptr::eq(self.arena, other.arena),
            "Values need to belong in the same Arena!"
        );
    }

    pub fn data(&self) -> f64 {
        self.arena.nodes.borrow()[self.idx].data
    }

    pub fn grad(&self) -> f64 {
        self.arena.nodes.borrow()[self.idx].grad
    }

    pub fn backward(&self) {
        let mut borrow = self.arena.nodes.borrow_mut();
        borrow.get_mut(self.idx).unwrap().grad = 1.0;
        for i in (0..=self.idx).rev() {
            let node = *borrow.get_mut(i).unwrap();

            match node.op {
                Op::Leaf => {}
                Op::Add(a, b) => {
                    borrow.get_mut(a).unwrap().grad += node.grad;
                    borrow.get_mut(b).unwrap().grad += node.grad;
                }
                Op::Mul(a, b) => {
                    let (a_data, b_data) =
                        (borrow.get(a).unwrap().data, borrow.get(b).unwrap().data);
                    borrow.get_mut(a).unwrap().grad += node.grad * b_data;
                    borrow.get_mut(b).unwrap().grad += node.grad * a_data;
                }
            }
        }
    }
}

impl Add for Value<'_> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        self.ensure_same_arena(&rhs);
        let res = self.data() + rhs.data();
        self.arena.push_node(res, Op::Add(self.idx, rhs.idx))
    }
}

impl Mul for Value<'_> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        self.ensure_same_arena(&rhs);
        let res = self.data() * rhs.data();
        self.arena.push_node(res, Op::Mul(self.idx, rhs.idx))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_value_data() {
        let arena = Arena::new();
        let a = arena.leaf(-4.0);
        let b = arena.leaf(2.0);
        assert_eq!(a.data(), -4.0);
        assert_eq!(b.data(), 2.0);
    }

    #[test]
    fn test_value_add() {
        let arena = Arena::new();
        let a = arena.leaf(-4.0);
        let b = arena.leaf(2.0);
        let c = a + b;
        assert_eq!(c.data(), -2.0);
    }

    #[test]
    #[should_panic(expected = "Values need to belong in the same Arena!")]
    fn test_value_add_diff_arena() {
        let arena1 = Arena::new();
        let arena2 = Arena::new();
        let a = arena1.leaf(1.0);
        let b = arena2.leaf(1.0);
        let _ = a + b;
    }

    #[test]
    fn test_arena_node_contains_op() {
        let arena = Arena::new();
        let a = arena.leaf(-4.0);
        let b = arena.leaf(2.0);
        let c = a + b;
        assert_eq!(a.arena.nodes.borrow()[a.idx].op, Op::Leaf);
        assert_eq!(c.arena.nodes.borrow()[c.idx].op, Op::Add(a.idx, b.idx));
    }

    #[test]
    fn test_value_backward_mul_distinct() {
        let arena = Arena::new();
        let a = arena.leaf(-4.0);
        let b = arena.leaf(2.0);
        let c = a * b;
        c.backward();
        assert_eq!((a.grad(), b.grad()), (2.0, -4.0));
    }

    #[test]
    fn test_value_backward_mul_reused() {
        let arena = Arena::new();
        let a = arena.leaf(-4.0);
        let b = a * a;
        b.backward();
        assert_eq!((a.grad(), b.grad()), (-8.0, 1.0));
    }

    #[test]
    fn test_value_backward_add_distinct() {
        let arena = Arena::new();
        let a = arena.leaf(-4.0);
        let b = arena.leaf(2.0);
        let c = a + b;
        c.backward();
        assert_eq!((a.grad(), b.grad(), c.grad()), (1.0, 1.0, 1.0));
    }

    #[test]
    fn test_value_backward_add_reused() {
        let arena = Arena::new();
        let a = arena.leaf(-4.0);
        let b = a + a;
        b.backward();
        assert_eq!((a.grad(), b.grad()), (2.0, 1.0));
    }

    #[test]
    fn test_arena_zero_grad() {
        let arena = Arena::new();
        let a = arena.leaf(-4.0);
        let b = arena.leaf(2.0);
        let c = a + b;
        c.backward();
        assert_eq!((a.grad(), b.grad(), c.grad()), (1.0, 1.0, 1.0));
        arena.zero_grad();
        assert_eq!((a.grad(), b.grad(), c.grad()), (0.0, 0.0, 0.0));
    }

    #[test]
    fn test_value_chain_rule() {
        let arena = Arena::new();
        let a = arena.leaf(-4.0);
        let b = arena.leaf(2.0);
        let c = (a * b) * b;
        c.backward();
        assert_eq!(c.data(), -16.0);
        assert_eq!(a.grad(), 4.0);
        assert_eq!(b.grad(), -16.0);
    }

    #[test]
    fn test_value_mul() {
        let arena = Arena::new();
        let a = arena.leaf(-4.0);
        let b = arena.leaf(2.0);
        let c = a * b;
        assert_eq!(c.data(), -8.0);
    }

    #[test]
    #[should_panic(expected = "Values need to belong in the same Arena!")]
    fn test_value_mul_diff_arena() {
        let arena1 = Arena::new();
        let arena2 = Arena::new();
        let a = arena1.leaf(1.0);
        let b = arena2.leaf(1.0);
        let _ = a * b;
    }

    #[test]
    fn test_value_mix_add_mul() {
        let arena = Arena::new();
        let a = arena.leaf(-4.0);
        let b = arena.leaf(2.0);
        let c = a * b + a;
        c.backward();
        assert_eq!((a.grad(), b.grad()), (3.0, -4.0));
    }
}
