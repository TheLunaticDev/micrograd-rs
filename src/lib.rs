use std::{cell::RefCell, ops::Add};

/// Struct that holds the value.
#[derive(Copy, Clone)]
struct Node {
    data: f64,
    grad: f64,
    op: Op,
}

/// Holds all operations supported by micrograd-rs.
#[derive(Debug, Copy, Clone)]
enum Op {
    Leaf,
    Add(usize, usize),
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
            }
        }
    }
}

impl Add for Value<'_> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let res = self.data() + rhs.data();
        self.arena.push_node(res, Op::Add(self.idx, rhs.idx))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::any::Any;

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
    fn test_arena_node_contains_op() {
        let arena = Arena::new();
        let a = arena.leaf(-4.0);
        let b = arena.leaf(2.0);
        let c = a + b;
        assert_eq!(
            a.arena.nodes.borrow()[c.idx].op.type_id(),
            Op::Leaf.type_id()
        );
        assert_eq!(
            b.arena.nodes.borrow()[c.idx].op.type_id(),
            Op::Leaf.type_id()
        );
        assert_eq!(
            c.arena.nodes.borrow()[c.idx].op.type_id(),
            Op::Add(0, 1).type_id()
        );
    }

    #[test]
    fn test_value_backward_add() {
        let arena = Arena::new();
        let a = arena.leaf(-4.0);
        let b = arena.leaf(-4.0);
        let c = a + b;
        c.backward();
        assert_eq!((a.grad(), b.grad(), c.grad()), (1.0, 1.0, 1.0));

        let arena = Arena::new();
        let a = arena.leaf(-4.0);
        let b = a + a;
        b.backward();
        assert_eq!(a.grad(), 2.0);
        assert_eq!(b.grad(), 1.0);
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
}
