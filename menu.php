<?php
session_start();
$title="";
error_reporting(0);
?><!doctype html>
<html lang="en">
<head>
	<meta charset="utf-8" />
	<link rel="icon" type="image/png" href="assets/img/favicon.ico">
	<meta http-equiv="X-UA-Compatible" content="IE=edge,chrome=1" />

	<title></title>

	<meta content='width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=0' name='viewport' />
    <meta name="viewport" content="width=device-width" />


    <!-- Bootstrap core CSS     -->
    <link href="../assets/css/bootstrap.min.css" rel="stylesheet" />

    <!-- Animation library for notifications   -->
    <link href="../assets/css/animate.min.css" rel="stylesheet"/>

    <!--  Light Bootstrap Table core CSS    -->
    <link href="../assets/css/light-bootstrap-dashboard.css" rel="stylesheet"/>


    <!--  CSS for Demo Purpose, don't include it in your project     -->
    <link href="../assets/css/demo.css" rel="stylesheet" />


    <!--     Fonts and icons     -->
    <link href="../common/font-awesome.min.css" rel="stylesheet">
    <link href='http://fonts.googleapis.com/css?family=Roboto:400,700,300' rel='stylesheet' type='text/css'>
    <link href="../assets/css/pe-icon-7-stroke.css" rel="stylesheet" />

</head>
<body>

<div class="wrapper">

<div class="sidebar" data-color="blue" data-image="../assets/img/sidebar-4.jpg">

    <!--

        Tip 1: you can change the color of the sidebar using: data-color="blue | azure | green | orange | red | purple"
        Tip 2: you can also add an image using data-image tag

    -->

    	<div class="sidebar-wrapper">
            <div class="logo">
                
                  <span style="font-size:20px;color:#FFF;margin-top:12px;"> Garbage Management <br><span style="font-size:12px;color:#000;"> 
                </span>
                </span>
               
            </div>

            <ul class="nav">
                
                
                <li class="active">
                    <a href="../dashboard/dashboard.php">
                        <i class="pe-7s-graph"></i>
                        <p>Home</p>
                    </a>
                </li>
                
                <li>
                    <a href="../user/select.php">
                        <i class="pe-7s-user"></i>
                        <p>Users</p>
                    </a>
                </li>
				
				 <li>
                    <a href="../agent/select.php">
                        <i class="pe-7s-user"></i>
                        <p>Agents</p>
                    </a>
                </li>
				
				 <li>
                    <a href="../bin/select.php">
                        <i class="pe-7s-user"></i>
                        <p>Bins</p>
                    </a>
                </li>
				
				 <li>
                    <a href="../complaints/select.php">
                        <i class="pe-7s-user"></i>
                        <p>Complaints</p>
                    </a>
                </li>
    
                
                 <li>
                    <a href="../payments/select.php">
                        <i class="pe-7s-news-paper"></i>
                        <p>Payments</p>
                    </a>
                </li>
                
                 <li>
                    <a href="../feedback/select.php">
                        <i class="pe-7s-news-paper"></i>
                        <p>Feedbacks </p>
                    </a>
                </li>
                
             
                
              
                 <li>
                    <a href="C:\xampp\htdocs\template\home.php">
                        <i class="pe-7s-bell"></i>
                        <p>Log Out</p>
                    </a>
                </li>
               
				
            </ul>
    	</div>
    </div>
    <div class="main-panel">
    
    
    
    <nav class="navbar navbar-default navbar-fixed">
            <div class="container-fluid">
                <div class="navbar-header">
                    <button type="button" class="navbar-toggle" data-toggle="collapse" data-target="#navigation-example-2">
                        <span class="sr-only">Toggle navigation</span>
                        <span class="icon-bar"></span>
                        <span class="icon-bar"></span>
                        <span class="icon-bar"></span>
                    </button>
                    <br>
                </div>
                <div class="collapse navbar-collapse">
                    <ul class="nav navbar-nav navbar-right">
                       <li class="dropdown">
                              <a href="#" class="dropdown-toggle" data-toggle="dropdown">
                                    <i class="fa fa-bell-o" style="font-size:30px;color:red"></i>
                                    <b class="caret hidden-sm hidden-xs"></b>
                                    
									<p class="hidden-lg hidden-md">
										5 Notifications
										<b class="caret"></b>
									</p>
                              </a>
                              <ul class="dropdown-menu">
							  <?php
								include("../connection.php");
								if($_SESSION['user']=='admin')
								{
								$sql = "SELECT * FROM user ORDER BY id DESC LIMIT 1";
								$result = mysqli_query($con,$sql);
								$rows = mysqli_num_rows($result);
								if ($rows > 0) {
									echo '<li><a href="../user/select.php">New user added !!!</a></li>';
								}
								}
							  ?>
							  <?php
								$sql = "SELECT * FROM complaints ORDER BY id DESC LIMIT 1";
								$result = mysqli_query($con,$sql);
								$rows = mysqli_num_rows($result);
								if ($rows > 0) {
									echo '<li><a href="../complaints/select.php">New complaints !!!</a></li>';
								}
							  ?>
                              </ul>
                        </li>
                    </ul>
                    <ul class="nav navbar-nav navbar-right">
                        <li>
                            <a >
                                <p><b>Welcome <?php echo $_SESSION['user']; ?></b></p>
                                <p>Username: <?php echo $_SESSION['name']; ?></p>
                            </a>
                        </li>
						<li class="separator hidden-lg hidden-md"></li>
                    </ul>
                </div>
            </div>
        </nav>