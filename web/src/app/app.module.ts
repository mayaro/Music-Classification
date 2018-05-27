import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';
import { HttpClientModule } from '@angular/common/http';
import { FormsModule } from '@angular/forms';
import { ToastrModule } from 'ngx-toastr';
import { NgxChartsModule } from '@swimlane/ngx-charts'

import { AppComponent } from './components/app.component';
import { HeaderComponent } from './components/header/header.component';
import { ClassificationComponent } from './components/classification/classification.component';
import { IterateGenresPipe } from './pipes/iterateGenres.pipe';
import { GenresChart } from './components/classification/chart/chart.component';

@NgModule({
  declarations: [
    AppComponent,
    HeaderComponent,
    ClassificationComponent,
    GenresChart,
    IterateGenresPipe,
  ],
  imports: [
    BrowserModule,
    BrowserAnimationsModule,
    ToastrModule.forRoot(),
    FormsModule,
    HttpClientModule,
    NgxChartsModule
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule { }
