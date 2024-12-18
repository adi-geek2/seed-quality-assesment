<?php
$quality = isset($_GET['quality']) ? htmlspecialchars($_GET['quality']) : 'Unknown';
?>
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Seed Quality Results</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
</head>
<body>
    <div class="container text-center mt-5">
        <h1>Seed Quality Results</h1>
        <div class="card mt-3">
            <div class="card-body">
                <h3>Quality: <span class="text-success"><?php echo $quality; ?></span></h3>
            </div>
        </div>
        <a href="index.html" class="btn btn-primary mt-4">Upload Another Image</a>
    </div>
</body>
</html>
