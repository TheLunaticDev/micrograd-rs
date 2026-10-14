use std::{cell::RefCell, ops::Add};

fn main() {
    let arena = Arena::new();
    let a = arena.leaf(-4.0);
    let b = arena.leaf(2.0);
    println!("{:.4}", a.data());
    println!("{:.4}", b.data());
    let c = a + b;
    println!("{:.4}", c.data());
    println!("{:?}", c.arena.nodes.borrow()[c.idx].op)
}

/// Struct that holds the value.
struct Node {
    data: f64,
    grad: f64,
    op: Op,
}

/// Holds all operations supported by micrograd-rs.
#[derive(Debug)]
enum Op {
    Leaf,
    Add(usize, usize),
}

/// A sharable object that holds nodes and allows operations.
struct Arena {
    nodes: RefCell<Vec<Node>>,
}

impl Arena {
    fn new() -> Self {
        Arena { nodes: RefCell::new(Vec::<Node>::new()) }
    }

    fn leaf(&self, data: f64) -> Value<'_> {
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
        Value {
            idx,
            arena: self,
        }
    }
}


#[derive(Clone, Copy)]
struct Value<'a> {
    idx: usize,
    arena: &'a Arena,
}

impl Value<'_> {
    fn data(&self) -> f64 {
        self.arena.nodes.borrow()[self.idx].data
    }
}

impl Add for Value<'_> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let res = self.data() + rhs.data();
        self.arena.push_node(res, Op::Add(self.idx, rhs.idx))
    }
}
