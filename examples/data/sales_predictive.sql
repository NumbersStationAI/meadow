-- Creating the tables

CREATE TABLE products (
    product_id INTEGER PRIMARY KEY,
    product_name TEXT NOT NULL,
    category_id INTEGER NOT NULL,
    price REAL NOT NULL,
    FOREIGN KEY (category_id) REFERENCES categories(category_id)
);

CREATE TABLE categories (
    category_id INTEGER PRIMARY KEY,
    category_name TEXT NOT NULL
);

CREATE TABLE sales (
    sale_id INTEGER PRIMARY KEY,
    product_id INTEGER NOT NULL,
    sale_date TEXT NOT NULL,
    sale_amount INTEGER NOT NULL,
    price REAL NOT NULL,
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);

CREATE TABLE comments (
    comment_id INTEGER PRIMARY KEY,
    product_id INTEGER NOT NULL,
    comment_text TEXT NOT NULL,
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);

-- Inserting sample data with some errors

INSERT INTO categories (category_id, category_name) VALUES
(1, 'Electronics'),
(2, 'Books'),
(3, 'Clothing'),
(4, 'Home & Kitchen');

INSERT INTO products (product_id, product_name, category_id, price) VALUES
(1, 'Smartphone', 1, 600.00),
(2, 'Laptop', 1, 900.00),
(3, 'Fiction Novel', 2, 10.00),
(4, 'Non-fiction Book', 2, 24.00),
(5, 'T-shirt', 3, 18.00),
(6, 'Jeans', 3, 30.00),
(7, 'Blender', 4, 40.00),
(8, 'Microwave', 4, 80.00),
(9, 'Headphones', 1, 50.00),
(10, 'Tablet', 1, 200.00);

INSERT INTO sales (sale_id, product_id, sale_date, sale_amount, price) VALUES
(1, 1, '2024-01-15', 10, 699.99),
(2, 1, '2024-01', 5, 699.99),
(3, 2, '2024-02-20', 3, 999.99),
(4, 3, '2024.03.05', 20, 19.99),
(5, 4, '2024/04/10', 7, 24.99),
(6, 5, '2024-05-15', 15, 14.99),
(7, 6, '2024.05', 10, 39.99),
(8, 7, '2024/06-25', 8, 49.99),
(9, 8, '2024.07/05', 6, 89.99),
(10, 9, '2024-08.10', 12, 59.99);

INSERT INTO comments (comment_id, product_id, comment_text) VALUES
(1, 1, 'Excellent product!'),
(2, 1, 'Could be better.'),
(3, 2, 'Very satisfied with my purchase.'),
(4, 3, 'A great read!'),
(5, 4, 'Informative and well-written.'),
(6, 5, 'Comfortable and affordable.'),
(7, 6, 'Great fit and quality.'),
(8, 7, 'Stopped working after a week.'),
(9, 8, 'Heats up food unevenly.'),
(10, 9, 'Sound quality is terrible.'),
(11, 10, 'Excellent tablet for the price.'),
(12, 10, 'Battery life is too short.'),
(13, 2, 'Overheats easily.'),
(14, 3, 'Pages fell out after a few reads.'),
(15, 5, 'Shrinks after wash.'),
(16, 6, 'Color faded quickly.'),
(17, 7, 'Blades are not sharp enough.'),
(18, 8, 'Makes a lot of noise.'),
(19, 9, 'Very comfortable to wear.'),
(20, 10, 'Screen resolution is excellent.');